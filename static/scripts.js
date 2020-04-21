function fileValidation(){
    var fileInput = document.getElementById('file');
    if (fileInput.files.length == 0) {
        alert('No file detected.');
        fileInput.value = '';
        document.getElementById('fileName').innerHTML = "Choose an Image";
        return false;
    }
    var filePath = fileInput.value;
    var allowedExtensions = /(\.jpg)$/i;
    if(!allowedExtensions.exec(filePath)){
        alert('Uploaded file is not .jpg');
        fileInput.value = '';
        document.getElementById('fileName').innerHTML = "Choose an Image";
        return false;
    }
    document.getElementById('fileName').innerHTML = "Image Selected";
    return true;
}