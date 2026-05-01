// ABOUTME: Tests for the tool functions and the agent's chat/loop behaviour.
// ABOUTME: Uses net/http/httptest to stand in for OpenRouter — fast and offline.

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "hello.txt")
	if err := os.WriteFile(path, []byte("hi"), 0644); err != nil {
		t.Fatal(err)
	}

	out, err := ReadFile(mustJSON(t, ReadFileInput{Path: path}))
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if out != "hi" {
		t.Errorf("got %q, want %q", out, "hi")
	}

	if _, err := ReadFile(mustJSON(t, ReadFileInput{Path: filepath.Join(dir, "missing.txt")})); err == nil {
		t.Error("expected error for missing file, got nil")
	}
}

func TestListFiles(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "a.txt"), "")
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0755); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, filepath.Join(dir, "sub", "b.txt"), "")

	out, err := ListFiles(mustJSON(t, ListFilesInput{Path: dir}))
	if err != nil {
		t.Fatal(err)
	}

	var got []string
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatal(err)
	}

	want := map[string]bool{"a.txt": true, "sub/": true, "sub/b.txt": true}
	if len(got) != len(want) {
		t.Errorf("got %d entries: %v, want %d: %v", len(got), got, len(want), want)
	}
	for _, name := range got {
		if !want[name] {
			t.Errorf("unexpected entry %q", name)
		}
	}
}

func TestEditFile_Replace(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	mustWrite(t, path, "hello world")

	if _, err := EditFile(mustJSON(t, EditFileInput{Path: path, OldStr: "hello", NewStr: "goodbye"})); err != nil {
		t.Fatalf("EditFile: %v", err)
	}

	got, _ := os.ReadFile(path)
	if string(got) != "goodbye world" {
		t.Errorf("got %q, want %q", got, "goodbye world")
	}
}

func TestEditFile_CreateNew(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "new.txt")

	if _, err := EditFile(mustJSON(t, EditFileInput{Path: path, OldStr: "", NewStr: "fresh"})); err != nil {
		t.Fatalf("EditFile: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read after create: %v", err)
	}
	if string(got) != "fresh" {
		t.Errorf("got %q, want %q", got, "fresh")
	}
}

func TestEditFile_OldStrNotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	mustWrite(t, path, "hello")

	if _, err := EditFile(mustJSON(t, EditFileInput{Path: path, OldStr: "nope", NewStr: "yep"})); err == nil {
		t.Error("expected error for missing old_str, got nil")
	}
}

func TestChat_SendsRequestAndParsesResponse(t *testing.T) {
	srv, reqs := fakeOpenRouter(t, []chatResponse{textResponse("hi back")})
	defer srv.Close()
	swapURL(t, srv.URL)

	msg, err := chat(context.Background(), "test-key", "test/model",
		[]Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if msg.Content != "hi back" {
		t.Errorf("got content %q, want %q", msg.Content, "hi back")
	}

	if len(*reqs) != 1 {
		t.Fatalf("got %d requests, want 1", len(*reqs))
	}
	got := (*reqs)[0]
	if got.Model != "test/model" {
		t.Errorf("model = %q, want %q", got.Model, "test/model")
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != "hi" {
		t.Errorf("unexpected messages: %+v", got.Messages)
	}
}

func TestChat_NonOKReturnsBodyInError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"bad model"}`, http.StatusNotFound)
	}))
	defer srv.Close()
	swapURL(t, srv.URL)

	_, err := chat(context.Background(), "k", "m", nil, nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "bad model") {
		t.Errorf("error should include body; got %q", err)
	}
}

func TestAgent_Run_ToolCallLoop(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "greeting.txt")
	mustWrite(t, path, "Hello, world!")

	args, _ := json.Marshal(EditFileInput{Path: path, OldStr: "Hello", NewStr: "Goodbye"})
	srv, reqs := fakeOpenRouter(t, []chatResponse{
		// First turn: model emits a tool call.
		toolCallResponse(ToolCall{
			ID: "call_1", Type: "function",
			Function: FunctionCall{Name: "edit_file", Arguments: string(args)},
		}),
		// Second turn (after tool result): final text.
		textResponse("done"),
	})
	defer srv.Close()
	swapURL(t, srv.URL)

	getUser := scriptedInput("please edit")
	agent := NewAgent("test-key", "test/model", getUser, []ToolDefinition{
		EditFileDefinition, ReadFileDefinition, ListFilesDefinition,
	})
	if err := agent.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "Goodbye, world!" {
		t.Errorf("file = %q, want %q", got, "Goodbye, world!")
	}

	if len(*reqs) != 2 {
		t.Fatalf("got %d chat calls, want 2", len(*reqs))
	}
	// Second request must contain the tool result keyed to call_1.
	var sawToolResult bool
	for _, m := range (*reqs)[1].Messages {
		if m.Role == "tool" && m.ToolCallID == "call_1" {
			sawToolResult = true
		}
	}
	if !sawToolResult {
		t.Errorf("second request missing tool-result message: %+v", (*reqs)[1].Messages)
	}
}

func TestAgent_Run_UnknownToolReportsError(t *testing.T) {
	srv, reqs := fakeOpenRouter(t, []chatResponse{
		toolCallResponse(ToolCall{
			ID: "call_x", Type: "function",
			Function: FunctionCall{Name: "no_such_tool", Arguments: "{}"},
		}),
		textResponse("ok"),
	})
	defer srv.Close()
	swapURL(t, srv.URL)

	agent := NewAgent("k", "m", scriptedInput("go"), []ToolDefinition{ReadFileDefinition})
	if err := agent.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	var toolMsg Message
	for _, m := range (*reqs)[1].Messages {
		if m.Role == "tool" {
			toolMsg = m
		}
	}
	if toolMsg.ToolCallID != "call_x" || !strings.Contains(toolMsg.Content, "tool not found") {
		t.Errorf("expected tool-not-found message, got %+v", toolMsg)
	}
}

// helpers

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func mustWrite(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(contents), 0644); err != nil {
		t.Fatal(err)
	}
}

func swapURL(t *testing.T, url string) {
	t.Helper()
	prev := openRouterURL
	openRouterURL = url
	t.Cleanup(func() { openRouterURL = prev })
}

func scriptedInput(inputs ...string) func() (string, bool) {
	i := 0
	return func() (string, bool) {
		if i >= len(inputs) {
			return "", false
		}
		s := inputs[i]
		i++
		return s, true
	}
}

func textResponse(s string) chatResponse {
	return chatResponse{Choices: []struct {
		Message Message `json:"message"`
	}{{Message: Message{Role: "assistant", Content: s}}}}
}

func toolCallResponse(calls ...ToolCall) chatResponse {
	return chatResponse{Choices: []struct {
		Message Message `json:"message"`
	}{{Message: Message{Role: "assistant", ToolCalls: calls}}}}
}

func fakeOpenRouter(t *testing.T, responses []chatResponse) (*httptest.Server, *[]chatRequest) {
	t.Helper()
	requests := []chatRequest{}
	i := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); !strings.HasPrefix(got, "Bearer ") {
			t.Errorf("missing Bearer auth header, got %q", got)
		}
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		requests = append(requests, req)

		if i >= len(responses) {
			t.Errorf("unexpected request #%d (only %d canned responses)", i+1, len(responses))
			http.Error(w, "no more responses", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(responses[i])
		i++
	}))
	return srv, &requests
}
