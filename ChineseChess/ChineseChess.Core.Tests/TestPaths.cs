namespace ChineseChess.Core.Tests;

internal static class TestPaths
{
    /// <summary>Walks up from the test assembly to the repository root (the folder holding artifacts/).</summary>
    public static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null)
            {
                if (Directory.Exists(Path.Combine(dir.FullName, "artifacts")) &&
                    File.Exists(Path.Combine(dir.FullName, "README.md")))
                {
                    return dir.FullName;
                }
                dir = dir.Parent;
            }
            throw new DirectoryNotFoundException("Could not locate repository root from " + AppContext.BaseDirectory);
        }
    }

    public static string DeployedOnnx => Path.Combine(RepoRoot, "artifacts", "cnn_policy_value.onnx");
}