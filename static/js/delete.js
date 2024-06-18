// 获取删除按钮和文件上传元素
const deleteButton = document.querySelector('.delete-file');
const fileUploader1 = document.getElementById('file-uploader');
const fileUploader2 = document.getElementById('feedback');

// 给删除按钮添加点击事件监听器
deleteButton.addEventListener('click', function() {
    // 清除文件上传元素的值
    fileUploader1.value = '';
    fileUploader2.innerHTML = '';
});
