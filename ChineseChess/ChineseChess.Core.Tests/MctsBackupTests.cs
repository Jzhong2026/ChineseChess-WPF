using ChineseChess.Models;
using ChineseChess.Services;
using Xunit;

namespace ChineseChess.Core.Tests;

/// <summary>
/// A2 — PUCT reads node.VisitCount at EVERY depth (SelectChild), so backup must
/// increment it on every node along the path, not just the root. Otherwise
/// sqrt(N_parent)==0 below the root and the exploration term vanishes.
/// Invariant under test: for every node, VisitCount == sum of its ChildN.
/// </summary>
public class MctsBackupTests
{
    private static readonly XiangqiEngine Engine = new();

    /// <summary>Builds root -> mid -> leaf[0..leafCount-1], all internal nodes expanded.</summary>
    private static (MctsNode Root, MctsNode Mid, MctsNode[] Leaves) Chain(int leafCount)
    {
        var root = Expand(Side.Red, 2, Side.Black);
        var mid = root.Children![0]!;
        Expand(mid, leafCount, Side.Red);
        return (root, mid, mid.Children!.Select(c => c!).ToArray());
    }

    private static MctsNode Expand(MctsNode node, int childCount, Side childSide)
    {
        node.Children = new MctsNode?[childCount];
        node.ChildMoves = new Move?[childCount];
        node.ChildN = new int[childCount];
        node.ChildW = new float[childCount];
        node.ChildP = new float[childCount];
        for (var i = 0; i < childCount; i++)
        {
            node.Children[i] = new MctsNode(node, 0.5f, Engine.EmptyBoard(), childSide);
            node.ChildP[i] = 0.5f;
        }
        node.IsLeaf = false;
        return node;
    }

    private static MctsNode Expand(Side side, int childCount, Side childSide) =>
        Expand(new MctsNode(null, 0f, Engine.EmptyBoard(), side), childCount, childSide);

    private static void AssertVisitConservation(MctsNode node)
    {
        if (node.IsLeaf) return;
        Assert.Equal(node.ChildN!.Sum(), node.VisitCount);
        foreach (var child in node.Children!)
        {
            if (child is not null) AssertVisitConservation(child);
        }
    }

    [Fact]
    public void BackupIncrementsVisitCountAtEveryDepth()
    {
        var (root, mid, leaves) = Chain(2);

        MctsAiService.BackupValue(leaves[0], new List<(MctsNode, int)> { (root, 0), (mid, 0) }, value: 1f);

        Assert.Equal(1, root.VisitCount);
        Assert.Equal(1, mid.VisitCount);
        Assert.Equal(1, leaves[0].VisitCount);
        AssertVisitConservation(root);
    }

    [Fact]
    public void BackupFlipsPerspectiveOncePerPly()
    {
        var (root, mid, leaves) = Chain(1);

        // Leaf is Red and reports +1 (Red winning). Black's move into it is -1,
        // Red's move into Black's node is +1 again.
        MctsAiService.BackupValue(leaves[0], new List<(MctsNode, int)> { (root, 0), (mid, 0) }, value: 1f);

        Assert.Equal(1f, root.ChildW![0]);
        Assert.Equal(-1f, mid.ChildW![0]);
    }

    [Fact]
    public void RepeatedBackupsAccumulateVisitsConsistently()
    {
        var (root, mid, leaves) = Chain(2);

        for (var i = 0; i < 5; i++)
            MctsAiService.BackupValue(leaves[0], new List<(MctsNode, int)> { (root, 0), (mid, 0) }, 1f);
        for (var i = 0; i < 3; i++)
            MctsAiService.BackupValue(leaves[1], new List<(MctsNode, int)> { (root, 0), (mid, 1) }, -1f);

        Assert.Equal(8, root.VisitCount);
        Assert.Equal(8, mid.VisitCount);
        Assert.Equal(5, leaves[0].VisitCount);
        Assert.Equal(3, leaves[1].VisitCount);
        Assert.Equal(-5f, mid.ChildW![0]);  // 5 leaf values of +1, flipped once at Black's node
        Assert.Equal(2f, root.ChildW![0]);  // +5 from the first batch, -3 from the second
        AssertVisitConservation(root);
    }

    [Fact]
    public void RootOnlyBackupStillCountsTheRoot()
    {
        var root = Expand(Side.Red, 2, Side.Black);
        var leaf = root.Children![0]!;

        MctsAiService.BackupValue(leaf, new List<(MctsNode, int)> { (root, 0) }, 1f);

        Assert.Equal(1, root.VisitCount);
        Assert.Equal(1, leaf.VisitCount);
        AssertVisitConservation(root);
    }
}