import React, { useState } from "react";
import axios from "axios";


function FileUploader(){
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState("");

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };
    
    const handleUpload = async () => {
        if (!selectedFile){
            setUploadStatus("No file selected");
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        //TODO: fetch file from backend using axios
        try{
            const response = await axios.post(
                "our backend point",
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }
            );

            setUploadStatus("File uploaded successfully!");
            console.log("Server response: ", response.data);
        } catch (error){
            setUploadStatus("Error uploading file.");
            console.error("Upload error: ", error);
        }
    };

  return (
	  <div>
	  <div className = "Title">
	  <p class="text-center">
	  <h1> ICDS Lidar Visualization </h1>
	  </p>
	  </div>


	  <div className="UploadSection">
	  <p class = "text-center">
	  <h2>File Upload</h2>
	  <input type="file" onChange={handleFileChange} />
	  <button>Process File</button>
	  {uploadStatus && <p>{uploadStatus}</p>}
	  </p>
	  </div>
	  </div>
);
  
}

export default FileUploader;
