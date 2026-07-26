# Specification for "LlamaPure" GitHub Actions CI/CD Workflow

## Objective
Generate a secure, robust GitHub Actions workflow file (`.github/workflows/publish.yml`) for the `LlamaPure` repository.
The workflow must trigger automatically when a new version tag (e.g., `v1.2.3`) is pushed. It must build the solution, run tests, pack the NuGet package, and upload it to NuGet.org.

---

## Strict Constraints for "act" (Local Testing) Compatibility
To ensure the workflow can be tested locally using `act` (nektos/act):
1. **Granular Steps**: Separate every logical phase into its own named `step`. Do not bundle building, testing, and packing into a single multi-line script.
2. **Environment Independent**: Use standard `dotnet` CLI commands instead of specialized OS-specific scripts where possible.
3. **Conditional Deployment**: The final NuGet upload step must check if it is running on a real tag push, but should allow `act` to run the rest of the steps safely without failing due to missing secrets or non-tag triggers.

---

## Workflow Requirements

### 1. Trigger Conditions
- Trigger the workflow when a tag matching the semantic versioning pattern `v*` is pushed (e.g., `v1.0.0`, `v2.1.3-beta`).
- Also allow manual triggers via `workflow_dispatch` for easier testing.

### 2. Job Structure
Create a single job named `build-test-publish` running on `ubuntu-latest`.

### 3. Step-by-Step Breakdown

Please implement the following exact sequence of steps:

- **Step 1: Checkout Source Code**
  - Use `actions/checkout@v4`.
  - Fetch all tags and history (set `fetch-depth: 0`) so that the versioning tool can read the git tag correctly.

- **Step 2: Setup .NET SDK**
  - Use `actions/setup-dotnet@v4`.
  - Configure it to use `.NET SDK 8.0` (or the latest stable SDK required to build .NET Standard 2.0 / 2.1 projects).

- **Step 3: Restore Dependencies**
  - Run `dotnet restore LlamaPure.sln`.

- **Step 4: Build Solution**
  - Run `dotnet build LlamaPure.sln --configuration Release --no-restore`.

- **Step 5: Run Unit/Integration Tests**
  - Run `dotnet test LlamaPure.Test/LlamaPure.Test.csproj --configuration Release --no-build`.
  - *Note for Agent*: Add a descriptive comment stating that native dependencies (`llama.dll` etc.) must be placed in the test output directory if the tests rely on the real native backend.

- **Step 6: Pack NuGet Package**
  - Run `dotnet pack LlamaPure/LlamaPure.csproj --configuration Release --no-build --output ./artifacts`.
  - Dynamically extract the version from the git tag if applicable, or rely on properties configured in the `.csproj`.

- **Step 7: Upload to NuGet.org (Conditional Deployment)**
  - Run `dotnet nuget push "./artifacts/*.nupkg" --api-key ${{ secrets.NUGET_API_KEY }} --source https://nuget.org --skip-duplicate`.
  - **Condition (`if`)**: This step must *only* execute if the trigger is a tag push (`github.ref_type == 'tag'`) AND the secret `NUGET_API_KEY` is not empty. This prevents `act` from failing locally due to a missing API key.

---

## Output Expectations
- Provide the complete, production-ready YAML syntax for `.github/workflows/publish.yml`.
- Include clear, professional YAML comments explaining what each step does.
- Ensure all action versions used (e.g., `@v4`) are modern and secure.
- Include a brief Markdown note at the end of your response explaining **how the user can test this locally using `act`** (e.g., the exact `act` command to mock a tag push or pass a dummy secret).
