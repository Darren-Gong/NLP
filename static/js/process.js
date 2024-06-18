function escapeHtml(html) {
    let text = document.createTextNode(html);
    let div = document.createElement('div');
    div.appendChild(text);
    return div.innerHTML;
}

function addRequestFile(fileName, fileContent) {
    var chatWindow = $('#chatWindow');
    $(".answer .tips").css({"display": "none"});    // 打赏卡隐藏
    let escapedMessage = escapeHtml(fileName);  // 对请求message进行转义，防止输入的是html而被浏览器渲染
    let requestMessageElement = $('<div class="message-bubble"><span class="chat-icon request-icon"></span><div class="message-text request"><p>' + escapedMessage + '</p></div></div>');
    chatWindow.append(requestMessageElement);
    let responseMessageElement = $('<div class="message-bubble"><span class="chat-icon response-icon"></span><div class="message-text response"><span class="loading-icon"><i class="fa fa-spinner fa-pulse fa-2x"></i></span></div></div>');
    chatWindow.append(responseMessageElement);
    chatWindow.scrollTop(chatWindow.prop('scrollHeight'));
}
// 处理文件上传后的操作
function handleFileUpload(file) {
    // 读取文件内容
    var reader = new FileReader();
    reader.onload = function (e) {
        var fileContent = e.target.result;

        // 添加请求消息到聊天窗口
        addRequestFile("上传文件内容: " + fileContent);

        // 在这里可以继续编写发送请求到服务器等操作
    };
    reader.readAsText(file);
}

// 点击 "Go !" 按钮时触发的事件
document.getElementById("chatBtn").addEventListener("click", function () {
    var fileInput = document.getElementById("file-uploader");
    var file = fileInput.files[0];

    if (file) {
        // 处理文件上传
        handleFileUpload(file);
    }
});


// %%%%%%%%%%%%
    var chatBtn = $('#chatBtn');
    var chatInput = $('#chatInput');
    var chatWindow = $('#chatWindow');

    // 存储对话信息,实现连续对话
    var messages = [];

    // 检查返回的信息是否是正确信息
    var resFlag = true

    // 创建自定义渲染器
    const renderer = new marked.Renderer();

    // 重写list方法
    renderer.list = function (body, ordered, start) {
        const type = ordered ? 'ol' : 'ul';
        const startAttr = (ordered && start) ? ` start="${start}"` : '';
        return `<${type}${startAttr}>\n${body}</${type}>\n`;
    };

    // 设置marked选项
    marked.setOptions({
        renderer: renderer,
        highlight: function (code, language) {
            const validLanguage = hljs.getLanguage(language) ? language : 'javascript';
            return hljs.highlight(code, {language: validLanguage}).value;
        }
    });
//%%%%%%%%%%%

// 添加响应消息到窗口,流式响应此方法会执行多次
function addResponseMessage(message) {
    let lastResponseElement = $(".message-bubble .response").last();
    lastResponseElement.empty();
    let escapedMessage;
    // 处理流式消息中的代码块
    let codeMarkCount = 0;
    let index = message.indexOf('```');
    while (index !== -1) {
        codeMarkCount++;
        index = message.indexOf('```', index + 3);
    }
    if (codeMarkCount % 2 === 1) {  // 有未闭合的 code
        escapedMessage = marked.parse(message + '\n\n```');
    } else if (codeMarkCount % 2 === 0 && codeMarkCount !== 0) {
        escapedMessage = marked.parse(message);  // 响应消息markdown实时转换为html
    } else if (codeMarkCount === 0) {  // 输出的代码没有markdown代码块
        if (message.includes('`')) {
            escapedMessage = marked.parse(message);  // 没有markdown代码块，但有代码段，依旧是markdown格式
        } else {
            escapedMessage = marked.parse(escapeHtml(message)); // 有可能不是markdown格式，都用escapeHtml处理后再转换，防止非markdown格式html紊乱页面
        }
    }
    lastResponseElement.append(escapedMessage);
    chatWindow.scrollTop(chatWindow.prop('scrollHeight'));
}



// 定义一个函数，用于处理服务器返回的数据并将结果显示在页面上
function handleResponse(data) {
    // 调用 addResponseMessage 函数将处理结果添加到聊天窗口
    addResponseMessage(data.result);
}

document.getElementById("chatBtn").addEventListener("click", function () {
    var selectedModel = document.getElementById("modelSelect").value;
    var inputText = document.getElementById("chatInput").value;

    // 检查输入文本是否为空
    if (inputText.trim() === "") {
        console.log("输入文本为空，不发送请求。");
        return; // 如果输入文本为空，则退出函数，不发送请求
    }

    fetch('/process_text', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({model: selectedModel, text: inputText}),
    })
    .then(response => response.json())
    .then(handleResponse) // 调用 handleResponse 函数处理服务器返回的数据
    .catch((error) => {
        console.error('Error:', error);
    });
});


// 处理用户上传文件并发送请求
document.getElementById("chatBtn").addEventListener("click", function () {
    var selectedModel = document.getElementById("modelSelect").value;
    var extraFunction = document.getElementById("extraFunctionSelect").value;
    var fileInput = document.getElementById("file-uploader");
    var file = fileInput.files[0];

    if (file) {
        var formData = new FormData();
        formData.append('file', file);
        formData.append("selected_model", selectedModel);
        formData.append("extra_function", extraFunction);
        fetch('/upload_file', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("chatWindow").innerHTML = "<h3>处理结果：</h3><p>" + data.result + "</p>";
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    }
});

// 在文本聚类选择器改变时的事件处理程序
$("#modelSelect").change(function() {
    // 获取选择的值
    var selectedModel = $(this).val();

    // 如果选择的是文本聚类
    if (selectedModel === "model4") {
        // 显示额外功能选择器
        $("#extraFunctionContainer").show();
    } else {
        // 隐藏额外功能选择器
        $("#extraFunctionContainer").hide();
    }
});


//
// // 处理用户输入并发送请求
// document.getElementById("chatBtn").addEventListener("click", function () {
//     var selectedModel = document.getElementById("modelSelect").value;
//     var inputText = document.getElementById("chatInput").value;
//
//     fetch('/process_text', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({model: selectedModel, text: inputText}),
//     })
//         .then(response => response.json())
//         .then(data => {
//             // 处理后端返回的数据，并将结果显示在页面上
//             document.getElementById("chatWindow").innerHTML = "<h3>处理结果：</h3><p>" + data.result + "</p>";
//         })
//         .catch((error) => {
//             console.error('Error:', error);
//         });
// });
//
// document.getElementById("chatBtn").addEventListener("click", function () {
//     var selectedModel = document.getElementById("modelSelect").value;
//     var fileInput = document.getElementById("file-uploader");
//     var file = fileInput.files[0];
//
//     if (file) {
//         var formData = new FormData();
//         formData.append('file', file);
//         formData.append("selected_model", selectedModel);
//
//         fetch('/upload_file', {
//             method: 'POST',
//             body: formData,
//         })
//         .then(response => response.json())
//         .then(data => {
//             document.getElementById("chatWindow").innerHTML = "<h3>处理结果：</h3><p>" + data.result + "</p>";
//         })
//         .catch((error) => {
//             console.error('Error:', error);
//         });
//     } else {
//         alert("请选择要上传的文件！");
//     }
// });
