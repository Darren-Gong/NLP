// 获取 select 元素
const selectElement = document.querySelector('.theme');

// 添加 change 事件监听器
selectElement.addEventListener('change', (event) => {
    // 当选择发生变化时，这个回调函数会被触发
    const selectedTheme = event.target.value; // 获取选择的主题值
    // 在这里执行相应的操作，比如根据选择的主题改变整体背景颜色
    changeBackgroundColor(selectedTheme);
});

// 定义一个函数来根据选择的主题改变整体背景颜色
function changeBackgroundColor(theme) {
    // 根据选择的主题值进行不同的操作
    switch (theme) {
        case 'light':
            document.body.style.backgroundColor = '#ffffff'; // 白色背景
            break;
        case 'gray':
            document.body.style.backgroundColor = '#f0f0f0'; // 灰色背景
            break;
        case 'light-red':
            document.body.style.backgroundColor = '#ffcccc'; // 浅红色背景
            break;
        case 'light-blue':
            document.body.style.backgroundColor = '#ccf2ff'; // 浅蓝色背景
            break;
        case 'light-purple':
            document.body.style.backgroundColor = '#f2ccff'; // 浅紫色背景
            break;
        case 'light-green':
            document.body.style.backgroundColor = '#ccffcc'; // 浅绿色背景
            break;
        case 'light-yellow':
            document.body.style.backgroundColor = '#ffffcc'; // 浅黄色背景
            break;

        // 添加其他主题的处理逻辑
        default:
            // 默认情况下，不做任何操作
            break;
    }
}
