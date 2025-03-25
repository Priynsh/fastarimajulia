using Documenter
using CustomARIMA  # Replace with your actual package name

makedocs(
    sitename = "CustomARIMA.jl",
    authors = "Priynsh",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",  # Enables pretty URLs on GitHub Actions
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/Priynsh/fastarimajulia.git",
    push_preview = true  # Enables preview for PRs
)
