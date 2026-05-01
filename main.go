// ABOUTME: Code-editing agent following Thorsten Ball's "How to Build an Agent" post.
// ABOUTME: Talks to OpenRouter's OpenAI-compatible chat completions API over plain HTTP.

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/joho/godotenv"
)

var openRouterURL = "https://openrouter.ai/api/v1/chat/completions"

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Tool struct {
	Type     string         `json:"type"`
	Function FunctionSchema `json:"function"`
}

type FunctionSchema struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type chatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
}

func chat(ctx context.Context, apiKey, model string, messages []Message, tools []Tool) (*Message, error) {
	body, err := json.Marshal(chatRequest{Model: model, Messages: messages, Tools: tools})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, openRouterURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openrouter %d: %s", resp.StatusCode, string(raw))
	}

	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		return nil, err
	}
	return &cr.Choices[0].Message, nil
}

type ToolDefinition struct {
	Name        string
	Description string
	InputSchema any
	Function    func(input json.RawMessage) (string, error)
}

func GenerateSchema[T any]() any {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	return reflector.Reflect(v)
}

type Agent struct {
	apiKey         string
	model          string
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
}

func NewAgent(apiKey, model string, getUserMessage func() (string, bool), tools []ToolDefinition) *Agent {
	return &Agent{apiKey: apiKey, model: model, getUserMessage: getUserMessage, tools: tools}
}

func (a *Agent) Run(ctx context.Context) error {
	var conversation []Message
	fmt.Printf("Chat with %s via OpenRouter (use 'ctrl-c' to quit)\n", a.model)

	tools := make([]Tool, 0, len(a.tools))
	for _, t := range a.tools {
		tools = append(tools, Tool{
			Type:     "function",
			Function: FunctionSchema{Name: t.Name, Description: t.Description, Parameters: t.InputSchema},
		})
	}

	readUserInput := true
	for {
		if readUserInput {
			fmt.Print("\x1b[94mYou\x1b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				return nil
			}
			conversation = append(conversation, Message{Role: "user", Content: userInput})
		}

		stop := startSpinner()
		msg, err := chat(ctx, a.apiKey, a.model, conversation, tools)
		stop()
		if err != nil {
			return err
		}
		conversation = append(conversation, *msg)

		if msg.Content != "" {
			fmt.Printf("\x1b[93mAgent\x1b[0m: %s\n", msg.Content)
		}

		if len(msg.ToolCalls) == 0 {
			readUserInput = true
			continue
		}

		for _, tc := range msg.ToolCalls {
			conversation = append(conversation, a.executeTool(tc))
		}
		readUserInput = false
	}
}

func (a *Agent) executeTool(tc ToolCall) Message {
	var def ToolDefinition
	var found bool
	for _, t := range a.tools {
		if t.Name == tc.Function.Name {
			def = t
			found = true
			break
		}
	}
	if !found {
		return Message{Role: "tool", ToolCallID: tc.ID, Content: "tool not found"}
	}

	fmt.Printf("\x1b[92mtool\x1b[0m: %s(%s)\n", tc.Function.Name, tc.Function.Arguments)
	out, err := def.Function(json.RawMessage(tc.Function.Arguments))
	if err != nil {
		return Message{Role: "tool", ToolCallID: tc.ID, Content: "error: " + err.Error()}
	}
	return Message{Role: "tool", ToolCallID: tc.ID, Content: out}
}

// startSpinner draws a Braille spinner while waiting on the model. Returns a
// stop function. No-op when stdout isn't a TTY (piped/redirected).
func startSpinner() func() {
	fi, _ := os.Stdout.Stat()
	if fi.Mode()&os.ModeCharDevice == 0 {
		return func() {}
	}
	frames := []rune("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		t := time.NewTicker(80 * time.Millisecond)
		defer t.Stop()
		for i := 0; ; i++ {
			fmt.Printf("\r\x1b[36m%c\x1b[0m thinking…", frames[i%len(frames)])
			select {
			case <-stop:
				fmt.Print("\r\x1b[K")
				return
			case <-t.C:
			}
		}
	}()
	return func() { close(stop); <-done }
}

var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
	InputSchema: GenerateSchema[ReadFileInput](),
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
}

func ReadFile(input json.RawMessage) (string, error) {
	in := ReadFileInput{}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	content, err := os.ReadFile(in.Path)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

var ListFilesDefinition = ToolDefinition{
	Name:        "list_files",
	Description: "List files and directories at a given path. If no path is provided, lists files in the current directory.",
	InputSchema: GenerateSchema[ListFilesInput](),
	Function:    ListFiles,
}

type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to current directory if not provided."`
}

func ListFiles(input json.RawMessage) (string, error) {
	in := ListFilesInput{}
	if len(input) > 0 {
		if err := json.Unmarshal(input, &in); err != nil {
			return "", err
		}
	}

	dir := "."
	if in.Path != "" {
		dir = in.Path
	}

	var files []string
	err := filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		if info.IsDir() {
			files = append(files, rel+"/")
		} else {
			files = append(files, rel)
		}
		return nil
	})
	if err != nil {
		return "", err
	}

	out, err := json.Marshal(files)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

var EditFileDefinition = ToolDefinition{
	Name: "edit_file",
	Description: `Make edits to a text file.

Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.
`,
	InputSchema: GenerateSchema[EditFileInput](),
	Function:    EditFile,
}

type EditFileInput struct {
	Path   string `json:"path" jsonschema_description:"The path to the file"`
	OldStr string `json:"old_str" jsonschema_description:"Text to search for - must match exactly and must only have one match exactly"`
	NewStr string `json:"new_str" jsonschema_description:"Text to replace old_str with"`
}

func EditFile(input json.RawMessage) (string, error) {
	in := EditFileInput{}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if in.Path == "" || in.OldStr == in.NewStr {
		return "", fmt.Errorf("invalid input parameters")
	}

	content, err := os.ReadFile(in.Path)
	if err != nil {
		if os.IsNotExist(err) && in.OldStr == "" {
			return createNewFile(in.Path, in.NewStr)
		}
		return "", err
	}

	oldContent := string(content)
	newContent := strings.Replace(oldContent, in.OldStr, in.NewStr, -1)
	if oldContent == newContent && in.OldStr != "" {
		return "", fmt.Errorf("old_str not found in file")
	}

	if err := os.WriteFile(in.Path, []byte(newContent), 0o644); err != nil {
		return "", err
	}
	return "OK", nil
}

func createNewFile(filePath, content string) (string, error) {
	dir := filepath.Dir(filePath)
	if dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}
	if err := os.WriteFile(filePath, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}
	return fmt.Sprintf("Successfully created file %s", filePath), nil
}

func main() {
	_ = godotenv.Load()

	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("OPENROUTER_API_KEY is not set (put it in .env or export it)")
		return
	}

	model := os.Getenv("OPENROUTER_MODEL")
	if model == "" {
		model = "z-ai/glm-4.5-air:free"
	}

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []ToolDefinition{ReadFileDefinition, ListFilesDefinition, EditFileDefinition}
	agent := NewAgent(apiKey, model, getUserMessage, tools)
	if err := agent.Run(context.Background()); err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}
