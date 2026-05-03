using System.Configuration;
using System.Data;
using System.Windows;

namespace ChineseChess
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public App()
        {
            new Bootstrapper(); // 👈 必须实例化
        }
    }

}
