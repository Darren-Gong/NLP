// 获取弹出窗口元素
var leftElement = document.querySelector('.left-element');

// 获取控制弹出窗口显示/隐藏的按钮（假设是一个按钮）
var toggleButton = document.getElementById('toggleButton');

// 添加点击事件监听器
toggleButton.addEventListener('click', function() {
        // 获取当前按钮的左侧偏移量
    var currentLeft = parseInt(toggleButton.style.left) || 0;
    var left_width=parseInt(window.getComputedStyle(leftElement).width) || 0;

    // // 将左侧偏移量增加 30px
    // var newLeft = currentLeft + 30;
    //
    // // 更新按钮的 left 属性
    // toggleButton.style.left = newLeft + 'px';
    // 检查弹出窗口的当前显示状态
    if (leftElement.style.display === 'none') {
        // 如果当前隐藏，则显示弹出窗口
        leftElement.style.display = 'block';
        var newLeft = currentLeft+left_width;
        toggleButton.style.left=newLeft+'px';
    } else {
        // 如果当前显示，则隐藏弹出窗口
        leftElement.style.display = 'none';
        toggleButton.style.left=10+'px';
    }
});

