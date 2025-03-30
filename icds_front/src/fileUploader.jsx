import React, { useState } from 'react';
import { Button, Form, Alert } from 'react-bootstrap';

const FileUploader = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [filePath, setFilePath] = useState('');

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }
  
    setUploading(true);
  
    try {
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate delay
  
      onUploadSuccess(file); // ✅ Send the real File object to App.js
      setError(null);
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };


  return (
    <div className="border rounded p-4 bg-light shadow-sm">
      <h4 className="mb-3">Upload Point Cloud Data</h4>
      
      {error && <Alert variant="danger">{error}</Alert>}
      
      <Form.Group className="mb-3">
        <Form.Label>Select a .pcd file</Form.Label>
        <Form.Control 
          type="file" 
          onChange={handleFileChange}
          accept=".pcd"
          disabled={uploading}
        />
      </Form.Group>
      
      <Button 
        variant="primary" 
        onClick={handleUpload}
        disabled={!file || uploading}
        className="w-100"
      >
        {uploading ? 'Uploading...' : 'Upload File'}
      </Button>
      
    </div>
  );
};

export default FileUploader;