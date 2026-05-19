"""Tests for regex-based code chunking (Go, TypeScript, C#)."""

import pytest
from mcp_code_rag.chunker import _chunk_go, _chunk_typescript, _chunk_csharp


def test_chunk_go_function():
    """Test chunking a Go function."""
    source = '''
func Add(a int, b int) int {
    return a + b
}
'''

    chunks = _chunk_go(source, "/test/math.go", "math")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "function"
    assert chunk.name == "Add"
    assert chunk.language == "go"
    assert "func Add" in chunk.signature


def test_chunk_go_method_with_receiver():
    """Test chunking a Go method with receiver."""
    source = '''
type Calculator struct {
    value int
}

func (c *Calculator) Add(x int) {
    c.value += x
}
'''

    chunks = _chunk_go(source, "/test/calc.go", "calc")

    # Should have type definition + method
    assert len(chunks) >= 2

    # Find the method
    method_chunk = None
    for chunk in chunks:
        if chunk.name == "Add":
            method_chunk = chunk
            break

    assert method_chunk is not None
    assert method_chunk.type == "method"
    assert method_chunk.metadata["is_method"] is True
    assert method_chunk.metadata["receiver_type"] == "*Calculator"


def test_chunk_go_type_struct():
    """Test chunking a Go struct type."""
    source = '''
type User struct {
    ID    int
    Name  string
    Email string
}
'''

    chunks = _chunk_go(source, "/test/models.go", "models")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "class"
    assert chunk.name == "User"
    assert "struct" in chunk.signature
    assert "User" in chunk.code


def test_chunk_go_type_interface():
    """Test chunking a Go interface type."""
    source = '''
type Reader interface {
    Read(p []byte) (n int, err error)
}
'''

    chunks = _chunk_go(source, "/test/interfaces.go", "interfaces")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "class"
    assert chunk.name == "Reader"
    assert "interface" in chunk.signature


def test_chunk_typescript_function():
    """Test chunking a TypeScript function."""
    source = '''
export function greet(name: string): string {
    return `Hello, ${name}!`;
}
'''

    chunks = _chunk_typescript(source, "/test/greet.ts", "typescript", "test")

    assert len(chunks) >= 1
    # Find the greet function
    greet_chunk = None
    for chunk in chunks:
        if chunk.name == "greet":
            greet_chunk = chunk
            break

    assert greet_chunk is not None
    assert greet_chunk.type == "function"
    assert greet_chunk.language == "typescript"


def test_chunk_typescript_async_function():
    """Test chunking a TypeScript async function."""
    source = '''
export async function fetchUser(id: number): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
}
'''

    chunks = _chunk_typescript(source, "/test/api.ts", "typescript", "api")

    # Find the fetchUser function
    fetch_chunk = None
    for chunk in chunks:
        if chunk.name == "fetchUser":
            fetch_chunk = chunk
            break

    assert fetch_chunk is not None
    assert fetch_chunk.type == "function"
    assert fetch_chunk.metadata.get("is_async") is True


def test_chunk_typescript_class():
    """Test chunking a TypeScript class."""
    source = '''
export class UserService {
    constructor(private api: Api) {}

    async getUser(id: number): Promise<User> {
        return this.api.get(`/users/${id}`);
    }

    async updateUser(id: number, data: Partial<User>): Promise<User> {
        return this.api.put(`/users/${id}`, data);
    }
}
'''

    chunks = _chunk_typescript(source, "/test/service.ts", "typescript", "services")

    # Should have class + methods
    assert len(chunks) >= 1
    class_chunk = chunks[0]
    assert class_chunk.type == "class"
    assert class_chunk.name == "UserService"


def test_chunk_csharp_class():
    """Test chunking a C# class."""
    source = '''
public class User
{
    public int Id { get; set; }
    public string Name { get; set; }
}
'''

    chunks = _chunk_csharp(source, "/test/User.cs", "User")

    assert len(chunks) >= 1
    class_chunk = chunks[0]
    assert class_chunk.type == "class"
    assert class_chunk.name == "User"
    assert class_chunk.language == "csharp"


def test_chunk_csharp_method():
    """Test chunking C# methods."""
    source = '''
public class Calculator
{
    public int Add(int a, int b)
    {
        return a + b;
    }

    public async Task<string> FetchData(string url)
    {
        var response = await http.GetAsync(url);
        return await response.Content.ReadAsStringAsync();
    }
}
'''

    chunks = _chunk_csharp(source, "/test/Calculator.cs", "Calculator")

    # Find the async method
    async_chunk = None
    for chunk in chunks:
        if chunk.name == "FetchData":
            async_chunk = chunk
            break

    assert async_chunk is not None
    assert async_chunk.type == "method"
    assert async_chunk.metadata.get("is_async") is True


def test_chunk_csharp_interface():
    """Test chunking a C# interface."""
    source = '''
public interface IRepository
{
    Task<T> GetAsync(int id);
    Task SaveAsync(T entity);
}
'''

    chunks = _chunk_csharp(source, "/test/IRepository.cs", "repositories")

    interface_chunk = None
    for chunk in chunks:
        if chunk.name == "IRepository":
            interface_chunk = chunk
            break

    assert interface_chunk is not None
    assert interface_chunk.type == "class"
    assert interface_chunk.metadata.get("class_kind") == "interface"


def test_chunk_go_without_receiver():
    """Test Go function without receiver."""
    source = '''
func main() {
    fmt.Println("Hello, world!")
}
'''

    chunks = _chunk_go(source, "/test/main.go", "main")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "function"
    assert chunk.metadata["is_method"] is False


def test_chunk_with_package():
    """Test that package is correctly extracted and used."""
    source = '''
package mypackage

func Process() {
    // logic
}
'''

    chunks = _chunk_go(source, "/test/process.go", "fallback")

    assert len(chunks) == 1
    chunk = chunks[0]
    # Should use detected package "mypackage", not fallback
    assert chunk.package == "mypackage"


# ── TypeScript additional cases (>= 5 total) ─────────────────────────────────


def test_chunk_typescript_class_with_extends():
    """Test chunking a TypeScript class that extends another class."""
    source = '''
export class AdminService extends UserService {
    deleteUser(id: number): void {
        // admin only
    }
}
'''

    chunks = _chunk_typescript(source, "/test/admin.ts", "typescript", "services")

    class_chunk = None
    for chunk in chunks:
        if chunk.name == "AdminService":
            class_chunk = chunk
            break

    assert class_chunk is not None
    assert class_chunk.type == "class"
    assert "extends" in class_chunk.signature
    assert "UserService" in class_chunk.signature


def test_chunk_typescript_plain_javascript_function():
    """Test chunking a plain (non-export) JavaScript function."""
    source = '''
function calculateSum(a, b) {
    return a + b;
}
'''

    chunks = _chunk_typescript(source, "/test/calc.js", "javascript", "calc")

    calc_chunk = None
    for chunk in chunks:
        if chunk.name == "calculateSum":
            calc_chunk = chunk
            break

    assert calc_chunk is not None
    assert calc_chunk.language == "javascript"
    assert calc_chunk.type == "function"
    assert calc_chunk.metadata.get("is_async") is False


# ── C# additional cases (>= 5 total) ─────────────────────────────────────────


def test_chunk_csharp_namespace_extraction():
    """Test that namespace is correctly extracted as the chunk package."""
    source = '''
namespace MyApp.Services
{
    public class AuthService
    {
        public void Login(string user) { }
    }
}
'''

    chunks = _chunk_csharp(source, "/test/AuthService.cs", "default_pkg")

    class_chunk = None
    for chunk in chunks:
        if chunk.name == "AuthService":
            class_chunk = chunk
            break

    assert class_chunk is not None
    assert class_chunk.package == "MyApp.Services"


def test_chunk_csharp_struct():
    """Test chunking a C# struct definition."""
    source = '''
public struct Point
{
    public int X { get; set; }
    public int Y { get; set; }
}
'''

    chunks = _chunk_csharp(source, "/test/Point.cs", "geometry")

    struct_chunk = None
    for chunk in chunks:
        if chunk.name == "Point":
            struct_chunk = chunk
            break

    assert struct_chunk is not None
    assert struct_chunk.type == "class"
    assert struct_chunk.metadata.get("class_kind") == "struct"
