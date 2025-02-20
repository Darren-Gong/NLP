// 功能
$(document).ready(function () {
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


    // 转义html代码(对应字符转移为html实体)，防止在浏览器渲染
    function escapeHtml(html) {
        let text = document.createTextNode(html);
        let div = document.createElement('div');
        div.appendChild(text);
        return div.innerHTML;
    }

    // 添加请求消息到窗口
    function addRequestMessage(message) {
        $(".answer .tips").css({"display": "none"});    // 打赏卡隐藏
        chatInput.val('');
        let escapedMessage = escapeHtml(message);  // 对请求message进行转义，防止输入的是html而被浏览器渲染
        let requestMessageElement = $('<div class="message-bubble"><span class="chat-icon request-icon"></span><div class="message-text request"><p>' + escapedMessage + '</p></div></div>');
        chatWindow.append(requestMessageElement);
        let responseMessageElement = $('<div class="message-bubble"><span class="chat-icon response-icon"></span><div class="message-text response"><span class="loading-icon"><i class="fa fa-spinner fa-pulse fa-2x"></i></span></div></div>');
        chatWindow.append(responseMessageElement);
        chatWindow.scrollTop(chatWindow.prop('scrollHeight'));
    }

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

    // 添加失败信息到窗口
    function addFailMessage(message) {
        let lastResponseElement = $(".message-bubble .response").last();
        lastResponseElement.empty();
        lastResponseElement.append('<p class="error">' + message + '</p>');
        chatWindow.scrollTop(chatWindow.prop('scrollHeight'));
        messages.pop() // 失败就让用户输入信息从数组删除
    }


    // 发送请求获得响应
    // async function sendRequest(data) {
    //     const response = await fetch(config.url, {
    //         method: 'POST',
    //         headers: {
    //             'Content-Type': 'application/json',
    //             'Authorization': 'Bearer ' + data.apiKey
    //         },
    //         body: JSON.stringify({
    //             "messages": data.prompts,
    //             "model": "gpt-3.5-turbo",
    //             "max_tokens": 1025,
    //             "temperature": 0.5,
    //             "top_p": 1,
    //             "n": 1,
    //             "stream": true
    //         })
    //     });

    //     const reader = response.body.getReader();
    //     let res = '';
    //     let str;
    //     while (true) {
    //         const {done, value} = await reader.read();
    //         if (done) {
    //             break;
    //         }
    //         str = '';
    //         res += new TextDecoder().decode(value).replace(/^data: /gm, '').replace("[DONE]", '');
    //         const lines = res.trim().split(/[\n]+(?=\{)/);
    //         for (let i = 0; i < lines.length; i++) {
    //             const line = lines[i];
    //             let jsonObj;
    //             try {
    //                 jsonObj = JSON.parse(line);
    //             } catch (e) {
    //                 break;
    //             }
    //             if (jsonObj.choices && jsonObj.choices[0].delta.content) {
    //                 str += jsonObj.choices[0].delta.content;
    //                 // addResponseMessage(str);
    //                 resFlag = true;
    //             } else {
    //                 if (jsonObj.error) {
    //                     // addFailMessage(jsonObj.error.type + " : " + jsonObj.error.message + jsonObj.error.code);
    //                     resFlag = false;
    //                 }
    //             }
    //         }
    //     }
    //     return str;
    // }

    // 处理用户输入
    chatBtn.click(function () {
        // 解绑键盘事件
        chatInput.off("keydown", handleEnter);

        // 保存api key与对话数据
        let data;
        if (config.apiKey !== '') {
            data = {"apiKey": atob(config.apiKey)};
        } else {
            data = {"apiKey": ""};
        }

        let apiKey = localStorage.getItem('apiKey');
        if (apiKey) {
            data.apiKey = apiKey;
        }

        let message = chatInput.val();
        if (message.length == 0) {
            // 重新绑定键盘事件
            chatInput.on("keydown", handleEnter);
            return
        }

        addRequestMessage(message);
        // 将用户消息保存到数组
        messages.push({"role": "user", "content": message})
        // // 收到回复前让按钮不可点击
        // chatBtn.attr('disabled', true)

        if (messages.length > 40) {
            // addFailMessage("此次对话长度过长，请点击下方删除按钮清除对话内容！");
            // 重新绑定键盘事件
            chatInput.on("keydown", handleEnter);
            chatBtn.attr('disabled', false) // 让按钮可点击
            return;
        }

        // 判读是否已开启连续对话
        if (localStorage.getItem('continuousDialogue') == 'true') {
            // 控制上下文，对话长度超过4轮，取最新的3轮,即数组最后7条数据
            data.prompts = messages.slice();  // 拷贝一份全局messages赋值给data.prompts,然后对data.prompts处理
            if (data.prompts.length > 8) {
                data.prompts.splice(0, data.prompts.length - 7);
            }
        } else {
            data.prompts = messages.slice();
            data.prompts.splice(0, data.prompts.length - 1); // 未开启连续对话，取最后一条
        }

        // sendRequest(data).then((res) => {
        //     chatInput.val('');
        //     // 收到回复，让按钮可点击
        //     chatBtn.attr('disabled', false)
        //     // 重新绑定键盘事件
        //     chatInput.on("keydown", handleEnter);
        //     // 判断是否是回复正确信息
        //     if (resFlag) {
        //         messages.push({"role": "assistant", "content": res});
        //         // 判断是否本地存储历史会话
        //         if (localStorage.getItem('archiveSession') == "true") {
        //             localStorage.setItem("session", JSON.stringify(messages));
        //         }
        //     }
        //     // 添加复制
        //     copy();
        // });
    });

    // Enter键盘事件
    function handleEnter(e) {
        if (e.keyCode == 13) {
            chatBtn.click();
            e.preventDefault();  //避免回车换行
        }
    }

    // 绑定Enter键盘事件
    chatInput.on("keydown", handleEnter);


    // 设置栏宽度自适应
    let width = $('.function .others').width();
    $('.function .settings .dropdown-menu').css('width', width);

    $(window).resize(function () {
        width = $('.function .others').width();
        $('.function .settings .dropdown-menu').css('width', width);
    });

    // apiKey输入框事件
    $(".settings-common .api-key").blur(function () {
        const apiKey = $(this).val();
        if (apiKey.length != 0) {
            localStorage.setItem('apiKey', apiKey);
        } else {
            localStorage.removeItem('apiKey');
        }
    })

    // 是否保存历史对话
    var archiveSession = localStorage.getItem('archiveSession');

    // 初始化archiveSession
    if (archiveSession == null) {
        archiveSession = "false";
        localStorage.setItem('archiveSession', archiveSession);
    }

    if (archiveSession == "true") {
        $("#chck-1").prop("checked", true);
    } else {
        $("#chck-1").prop("checked", false);
    }

    $('#chck-1').click(function () {
        if ($(this).prop('checked')) {
            // 开启状态的操作
            localStorage.setItem('archiveSession', true);
            if (messages.length != 0) {
                localStorage.setItem("session", JSON.stringify(messages));
            }
        } else {
            // 关闭状态的操作
            localStorage.setItem('archiveSession', false);
            localStorage.removeItem("session");
        }
    });

    // 加载历史保存会话
    if (archiveSession == "true") {
        const messagesList = JSON.parse(localStorage.getItem("session"));
        if (messagesList) {
            messages = messagesList;
            $.each(messages, function (index, item) {
                if (item.role === 'user') {
                    addRequestMessage(item.content)
                } else if (item.role === 'assistant') {
                    // addResponseMessage(item.content)
                }
            });
            // 添加复制
            copy();
        }
    }

    // 是否连续对话
    var continuousDialogue = localStorage.getItem('continuousDialogue');

    // 初始化continuousDialogue
    if (continuousDialogue == null) {
        continuousDialogue = "true";
        localStorage.setItem('continuousDialogue', continuousDialogue);
    }

    if (continuousDialogue == "true") {
        $("#chck-2").prop("checked", true);
    } else {
        $("#chck-2").prop("checked", false);
    }

    $('#chck-2').click(function () {
        if ($(this).prop('checked')) {
            localStorage.setItem('continuousDialogue', true);
        } else {
            localStorage.setItem('continuousDialogue', false);
        }
    });

    // 删除功能
    $(".delete a").click(function () {
        chatWindow.empty();
        $(".answer .tips").css({"display": "flex"});
        messages = [];
        localStorage.removeItem("session");
    });

    // 截图功能
    $(".screenshot a").click(function () {
        // 创建副本元素
        const clonedChatWindow = chatWindow.clone();
        clonedChatWindow.css({
            position: "absolute",
            left: "-9999px",
            overflow: "visible",
            width: chatWindow.width(),
            height: "auto"
        });
        $("body").append(clonedChatWindow);
        // 截图
        html2canvas(clonedChatWindow[0], {
            allowTaint: false,
            useCORS: true,
            scrollY: 0,
        }).then(function (canvas) {
            // 将 canvas 转换成图片
            const imgData = canvas.toDataURL('image/png');
            // 创建下载链接
            const link = document.createElement('a');
            link.download = "screenshot_" + Math.floor(Date.now() / 1000) + ".png";
            link.href = imgData;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            clonedChatWindow.remove();
        });
    });

    // 复制代码功能
    function copy() {
        $('pre').each(function () {
            let btn = $('<button class="copy-btn">复制</button>');
            $(this).append(btn);
            btn.hide();
        });

        $('pre').hover(
            function () {
                $(this).find('.copy-btn').show();
            },
            function () {
                $(this).find('.copy-btn').hide();
            }
        );

        $('pre').on('click', '.copy-btn', function () {
            let text = $(this).siblings('code').text();
            // 创建一个临时的 textarea 元素
            let textArea = document.createElement("textarea");
            textArea.value = text;
            document.body.appendChild(textArea);

            // 选择 textarea 中的文本
            textArea.select();

            // 执行复制命令
            try {
                document.execCommand('copy');
                $(this).text('复制成功');
            } catch (e) {
                $(this).text('复制失败');
            }

            // 从文档中删除临时的 textarea 元素
            document.body.removeChild(textArea);

            setTimeout(() => {
                $(this).text('复制');
            }, 2000);
        });
    }

    // // 禁用右键菜单
    // document.addEventListener('contextmenu', function (e) {
    //     e.preventDefault();  // 阻止默认事件
    // });

    // 禁止键盘F12键
    // document.addEventListener('keydown', function (e) {
    //     if (e.key == 'F12') {
    //         e.preventDefault(); // 如果按下键F12,阻止事件
    //     }
    // });
});

const fileUploader = document.getElementById('file-uploader');

fileUploader.addEventListener('change', (event) => {
    const files = event.target.files;
    console.log('files', files);
    const feedback = document.getElementById('feedback'); // 确保使用的是同一个变量
    const msg = `${files[0].name} 上传成功!`;
    feedback.textContent = msg;

    // 读取文件内容并保存
    const reader = new FileReader();

    reader.onload = function () {
        const fileContent = reader.result;
        console.log('File content:', fileContent);

        // 在这里处理文件内容，你可以将其保存到服务器或者进行其他操作
        // 例如，你可以使用 Ajax 将文件内容发送到服务器
        localStorage.setItem(files[0].name, fileContent);

        // 调用 addBorder 函数来添加边框
        addBorder(feedback);
    };
    reader.readAsText(files[0]);
});

function addBorder(feedbackElement) {
  feedbackElement.classList.add('border');
  setTimeout(() => removeBorder(feedbackElement), 2000);
}

function removeBorder(feedbackElement) {
  feedbackElement.classList.remove('border');
  // 清空文本内容
  feedbackElement.textContent = '';
}