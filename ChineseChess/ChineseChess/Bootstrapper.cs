using System;
using System.Collections.Generic;
using System.Windows;
using Caliburn.Micro;
using ChineseChess.Services;
using ChineseChess.ViewModels;

namespace ChineseChess;

public sealed class Bootstrapper : BootstrapperBase
{
    private SimpleContainer _container = null!;

    public Bootstrapper()
    {
        Initialize();
    }

    protected override void Configure()
    {
        _container = new SimpleContainer();
        _container.Instance(_container);
        _container.Singleton<IWindowManager, WindowManager>();
        _container.Singleton<IEventAggregator, EventAggregator>();
        _container.Singleton<XiangqiEngine>();
        _container.Singleton<XiangqiAiService>();
        _container.Singleton<SoundService>();
        _container.PerRequest<BoardViewModel>();
        _container.PerRequest<SidePanelViewModel>();
        _container.PerRequest<ShellViewModel>();
    }

    protected override object GetInstance(Type service, string key) => _container.GetInstance(service, key);

    protected override IEnumerable<object> GetAllInstances(Type service) => _container.GetAllInstances(service);

    protected override void BuildUp(object instance) => _container.BuildUp(instance);

    protected override async void OnStartup(object sender, StartupEventArgs e)
    {
        await DisplayRootViewForAsync<ShellViewModel>();
    }
}
