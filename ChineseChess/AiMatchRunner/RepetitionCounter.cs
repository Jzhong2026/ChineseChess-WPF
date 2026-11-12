namespace ChineseChess.AiMatchRunner;

/// <summary>
/// Counts occurrences of each position so a draw is declared when a position has
/// actually occurred three times. Counting consecutive re-visits instead would
/// both invent draws on short cycles and miss genuine interleaved repetitions.
/// </summary>
public sealed class RepetitionCounter
{
    private readonly Dictionary<string, int> _counts = new();

    public int MaxOccurrences { get; private set; }

    /// <summary>Records a position and returns true once it has occurred three times.</summary>
    public bool Add(string positionKey)
    {
        var count = _counts.TryGetValue(positionKey, out var seen) ? seen + 1 : 1;
        _counts[positionKey] = count;
        if (count > MaxOccurrences) MaxOccurrences = count;
        return count >= 3;
    }
}