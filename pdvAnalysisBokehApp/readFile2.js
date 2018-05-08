//Javascript to read in file as a text and pass to Python for processing

// Define file based in HTML5 file input tag
var file = document.getElementById('upload').files[0];

// Send post request to upload file
FileUpload(file);

function FileUpload(file) {
  var reader = new FileReader();  

  reader.readAsText(file);
  document.getElementById("modelid_uploadButton").childNodes[0].setAttribute("class", "bk-bs-btn bk-bs-btn-warning")
  $('button').


  reader.onload = function(e) {
    //console.log(reader.result)
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:15000/", true);
    xhr.setRequestHeader('Content-Type', 'text/plain');
    xhr.send(reader.result);

    // Change button color to green
    document.getElementById("modelid_uploadButton").childNodes[0].setAttribute("class", "bk-bs-btn bk-bs-btn-success")

    // Show preview of first 10 rows of file
    //filePreview = document.getElementById("modelid_filePreview").childNodes[0].childNodes[0].innerText
    //filePreview = reader.result.split('\n').slice(0,10);
    //document.getElementById("modelid_filePreview").childNodes[0].childNodes[0].innerHTML = "<h1> File Preview </h1>" + reader.result.split('\n').slice(0,10);
    document.getElementById("fileContents").innerText = reader.result.split('\n').slice(0,10);
  }
}
