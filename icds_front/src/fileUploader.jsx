import React, { useState } from "react";
import axios from "axios";
import { ProgressBar, Alert, Card, Button } from "react-bootstrap";

function FileUploader() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState("");
    const [uploadProgress, setUploadProgress] = useState(0);
    const [isUploading, setIsUploading] = useState(false);
    const [previewUrl, setPreviewUrl] = useState("");

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setUploadStatus("");
        
        // Create preview for images
        if (file && file.type.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = (e) => setPreviewUrl(e.target.result);
            reader.readAsDataURL(file);
        } else {
            setPreviewUrl("");
        }
    };
    
    const handleUpload = async () => {
        if (!selectedFile) {
            setUploadStatus("Please select a file first");
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        setIsUploading(true);
        setUploadProgress(0);
        
        try {
            const response = await axios.post(
                "your-backend-endpoint",
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                    onUploadProgress: (progressEvent) => {
                        const progress = Math.round(
                            (progressEvent.loaded * 100) / progressEvent.total
                        );
                        setUploadProgress(progress);
                    }
                }
            );

            setUploadStatus("success");
            console.log("Server response: ", response.data);
        } catch (error) {
            setUploadStatus("error");
            console.error("Upload error: ", error);
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <Card className="shadow-sm mt-4">
            <Card.Body>
                <Card.Title className="mb-4">
                    <i className="bi bi-cloud-arrow-up me-2"></i>
                    File Upload
                </Card.Title>
                
                <div className="mb-3">
                    <label className="form-label">Select a file to upload:</label>
                    <input 
                        type="file" 
                        onChange={handleFileChange}
                        className="form-control"
                        disabled={isUploading}
                    />
                </div>
                
                {previewUrl && (
                    <div className="mb-3 text-center">
                        <img 
                            src={previewUrl} 
                            alt="Preview" 
                            className="img-thumbnail" 
                            style={{ maxHeight: "200px" }}
                        />
                    </div>
                )}
                
                <div className="d-flex justify-content-between align-items-center mb-3">
                    <Button 
                        variant="primary" 
                        onClick={handleUpload}
                        disabled={!selectedFile || isUploading}
                    >
                        {isUploading ? (
                            <>
                                <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                                Uploading...
                            </>
                        ) : "Upload File"}
                    </Button>
                    
                    {selectedFile && !isUploading && (
                        <div className="text-muted">
                            <small>
                                {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)
                            </small>
                        </div>
                    )}
                </div>
                
                {isUploading && (
                    <ProgressBar 
                        now={uploadProgress} 
                        label={`${uploadProgress}%`} 
                        animated 
                        className="mb-3"
                    />
                )}
                
                {uploadStatus === "success" && (
                    <Alert variant="success" className="mt-3">
                        <i className="bi bi-check-circle-fill me-2"></i>
                        File uploaded successfully!
                    </Alert>
                )}
                
                {uploadStatus === "error" && (
                    <Alert variant="danger" className="mt-3">
                        <i className="bi bi-exclamation-triangle-fill me-2"></i>
                        Error uploading file. Please try again.
                    </Alert>
                )}
            </Card.Body>
            
            <Card.Footer className="text-muted small">
                Supported formats: .jpg, .png, .pdf, .pcd (max 10MB)
            </Card.Footer>
        </Card>
    );
}

export default FileUploader;