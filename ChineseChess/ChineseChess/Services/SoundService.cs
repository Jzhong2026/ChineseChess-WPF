using System.Media;
using System.IO;
using ChineseChess.Models;

namespace ChineseChess.Services;

public sealed class SoundService
{
    private readonly SoundPlayer? _movePlayer;
    private readonly SoundPlayer? _checkPlayer;

    public SoundService()
    {
        var baseDirectory = AppContext.BaseDirectory;
        _movePlayer = CreatePlayer(Path.Combine(baseDirectory, "Assets", "Sounds", "move.wav"));
        _checkPlayer = CreatePlayer(Path.Combine(baseDirectory, "Assets", "Sounds", "check.wav"));
    }

    public void PlayForStatus(GameStatus status)
    {
        _movePlayer?.Play();
        if (status is GameStatus.RedCheck or GameStatus.BlackCheck)
        {
            Task.Delay(180).ContinueWith(_ => _checkPlayer?.Play(), TaskScheduler.Default);
        }
    }

    private static SoundPlayer? CreatePlayer(string path)
    {
        if (!File.Exists(path)) return null;
        var player = new SoundPlayer(path);
        player.LoadAsync();
        return player;
    }
}
