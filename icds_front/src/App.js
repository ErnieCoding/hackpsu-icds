import React, { useState } from 'react';
import './styles/App.css';
import FileUploader from './fileUploader';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Container, Row, Col, Card, ListGroup, Badge } from 'react-bootstrap';

function App() {
  const [pointCloudData, setPointCloudData] = useState(null);
  const [identifiedObjects, setIdentifiedObjects] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);

  // Function to send point cloud data to Django backend
  const sendToBackend = async (filePath) => {
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/process-point-cloud/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          is_cie_data: false,
          is_pcd: true,
          file_path: filePath
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        // Process the response data
        const processedData = processResponseData(data);
        setPointCloudData(processedData);
        
        // Detect objects in the point cloud
        const objects = detectObjects(processedData.points);
        setIdentifiedObjects(objects);
        setSelectedObject(objects.length > 0 ? objects[0] : null);
        
        setError(null);
      } else {
        throw new Error(data.message || 'Unknown error occurred');
      }
    } catch (err) {
      setError(`Failed to process point cloud data: ${err.message}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // Process the response data from the backend
  const processResponseData = (responseData) => {
    const { points, colors } = responseData;
    
    // Convert points array (which contains [x,y,z] arrays) into array of point objects
    const pointObjects = points.map((point, index) => {
      return {
        x: point[0],
        y: point[1],
        z: point[2],
        r: colors ? Math.floor(colors[index][0] * 255) : Math.floor(Math.random() * 255),
        g: colors ? Math.floor(colors[index][1] * 255) : Math.floor(Math.random() * 255),
        b: colors ? Math.floor(colors[index][2] * 255) : Math.floor(Math.random() * 255)
      };
    });
    
    // Calculate bounds for the point cloud
    const bounds = calculateBounds(points);
    
    return {
      points: pointObjects,
      metadata: {
        pointCount: points.length,
        bounds: bounds
      }
    };
  };

  // Calculate bounds for the point cloud
  const calculateBounds = (points) => {
    if (!points || points.length === 0) {
      return { x: { min: 0, max: 0 }, y: { min: 0, max: 0 }, z: { min: 0, max: 0 } };
    }
    
    const bounds = {
      x: { min: Infinity, max: -Infinity },
      y: { min: Infinity, max: -Infinity },
      z: { min: Infinity, max: -Infinity }
    };
    
    points.forEach(point => {
      bounds.x.min = Math.min(bounds.x.min, point[0]);
      bounds.x.max = Math.max(bounds.x.max, point[0]);
      bounds.y.min = Math.min(bounds.y.min, point[1]);
      bounds.y.max = Math.max(bounds.y.max, point[1]);
      bounds.z.min = Math.min(bounds.z.min, point[2]);
      bounds.z.max = Math.max(bounds.z.max, point[2]);
    });
    
    return bounds;
  };

  // Mock function to simulate object detection
  const detectObjects = (pointData) => {
    // In a real app, this would come from your ML model/backend
    const mockObjects = [
      {
        id: 1,
        type: 'chair',
        confidence: 0.92,
        bounds: {
          x: { min: 1.2, max: 2.5 },
          y: { min: 0.8, max: 1.9 },
          z: { min: 0.1, max: 0.8 }
        },
        pointCount: 1245
      },
      {
        id: 2,
        type: 'table',
        confidence: 0.87,
        bounds: {
          x: { min: 3.0, max: 5.2 },
          y: { min: 2.1, max: 4.0 },
          z: { min: 0.7, max: 0.8 }
        },
        pointCount: 3560
      },
      {
        id: 3,
        type: 'human',
        confidence: 0.78,
        bounds: {
          x: { min: 6.1, max: 7.0 },
          y: { min: 1.5, max: 2.3 },
          z: { min: 0.0, max: 1.8 }
        },
        pointCount: 2890
      }
    ];
    
    return mockObjects.filter(obj => obj.confidence > 0.75);
  };

  const handleDataReceived = (data) => {
    // If data is a file path string, send it to the backend
    if (typeof data === 'string') {
      sendToBackend(data);
    } else {
      // For backward compatibility or local testing
      setIsLoading(true);
      try {
        const processedData = processPointCloudData(data);
        setPointCloudData(processedData);
        
        // Detect objects in the point cloud
        const objects = detectObjects(processedData.points);
        setIdentifiedObjects(objects);
        setSelectedObject(objects.length > 0 ? objects[0] : null);
        
        setError(null);
      } catch (err) {
        setError('Failed to process point cloud data');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    }
  };

  // Legacy function for local processing
  const processPointCloudData = (rawData) => {
    return {
      points: Array(100).fill(0).map((_, i) => ({
        x: Math.random() * 10,
        y: Math.random() * 10,
        z: Math.random() * 10,
        r: Math.floor(Math.random() * 255),
        g: Math.floor(Math.random() * 255),
        b: Math.floor(Math.random() * 255)
      })),
      metadata: {
        pointCount: 100,
        bounds: {
          x: { min: 0, max: 10 },
          y: { min: 0, max: 10 },
          z: { min: 0, max: 10 }
        }
      }
    };
  };

  // Handler for selecting an object from the list
  const handleObjectSelect = (object) => {
    setSelectedObject(object);
  };

  return (
    <div className="App">
      <Container fluid className="p-5 bg-primary text-white text-center">
        <h1>3D Point Cloud Analyzer</h1>
        <p>Upload a file to identify objects in 3D space</p>
      </Container>

      <Container className="mt-5">
        <Row>
          <Col md={8}>
            <FileUploader onUploadSuccess={handleDataReceived} />
            
            {/* Point Cloud Data Display */}
            <Card className="mt-4 shadow">
              <Card.Header className="bg-dark text-white">
                <h4 className="mb-0">Point Cloud Data</h4>
              </Card.Header>
              <Card.Body style={{ minHeight: '300px' }}>
                {isLoading ? (
                  <div className="text-center py-4">
                    <div className="spinner-border text-primary" role="status"></div>
                    <p className="mt-2">Analyzing point cloud...</p>
                  </div>
                ) : error ? (
                  <div className="text-center py-4 text-danger">
                    <i className="bi bi-exclamation-triangle" style={{ fontSize: '2rem' }}></i>
                    <p className="mt-2">{error}</p>
                  </div>
                ) : pointCloudData ? (
                  <>
                    <div className="border rounded p-3 bg-light mb-3" style={{ height: '200px', overflow: 'auto' }}>
                      <h6>Point Data Sample:</h6>
                      <pre className="mb-0 small">
                        {JSON.stringify(pointCloudData.points.slice(0, 20), null, 2)}
                      </pre>
                    </div>
                    <div className="d-flex justify-content-between">
                      <span>Total Points: <strong>{pointCloudData.metadata.pointCount.toLocaleString()}</strong></span>
                      <button className="btn btn-sm btn-outline-primary">
                        View Full 3D Visualization
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-4 text-muted">
                    <i className="bi bi-cloud-upload" style={{ fontSize: '2rem' }}></i>
                    <p className="mt-2">Upload a file to view point cloud data</p>
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>

          {/* Identified Objects Panel */}
          <Col md={4}>
            <Card className="shadow">
              <Card.Header className="bg-success text-white">
                <h4 className="mb-0">Identified Objects</h4>
              </Card.Header>
              <Card.Body>
                {isLoading ? (
                  <div className="text-center py-4">
                    <div className="spinner-border spinner-border-sm text-success" role="status"></div>
                    <p className="mt-2">Detecting objects...</p>
                  </div>
                ) : identifiedObjects.length > 0 ? (
                  <ListGroup variant="flush">
                    {identifiedObjects.map(obj => (
                      <ListGroup.Item 
                        key={obj.id} 
                        className="d-flex justify-content-between align-items-start"
                        action
                        active={selectedObject && selectedObject.id === obj.id}
                        onClick={() => handleObjectSelect(obj)}
                      >
                        <div>
                          <h6 className="mb-1 text-capitalize">{obj.type}</h6>
                          <small className={selectedObject && selectedObject.id === obj.id ? "text-white" : "text-muted"}>
                            Points: {obj.pointCount} | 
                            Confidence: {(obj.confidence * 100).toFixed(0)}%
                          </small>
                        </div>
                        <Badge bg={selectedObject && selectedObject.id === obj.id ? "light" : "success"} 
                               text={selectedObject && selectedObject.id === obj.id ? "dark" : "white"} 
                               pill>
                          {obj.type === 'human' ? 'Person' : obj.type}
                        </Badge>
                      </ListGroup.Item>
                    ))}
                  </ListGroup>
                ) : pointCloudData ? (
                  <div className="text-center py-4 text-muted">
                    <i className="bi bi-exclamation-triangle" style={{ fontSize: '2rem' }}></i>
                    <p className="mt-2">No objects detected</p>
                  </div>
                ) : (
                  <div className="text-center py-4 text-muted">
                    <i className="bi bi-box-seam" style={{ fontSize: '2rem' }}></i>
                    <p className="mt-2">Objects will appear here after analysis</p>
                  </div>
                )}
              </Card.Body>
              {identifiedObjects.length > 0 && (
                <Card.Footer className="small text-muted">
                  Detected {identifiedObjects.length} object{identifiedObjects.length !== 1 ? 's' : ''}
                </Card.Footer>
              )}
            </Card>

            {/* Object Details Panel (appears when objects are detected) */}
            {selectedObject && (
              <Card className="mt-3 shadow-sm">
                <Card.Header className="bg-light">
                  <h5 className="mb-0">Object Details</h5>
                </Card.Header>
                <Card.Body>
                  <div className="mb-3">
                    <h6>Selected: <span className="text-capitalize">{selectedObject.type}</span></h6>
                    <div className="small">
                      <div>Dimensions:</div>
                      <ul className="mb-1">
                        <li>Width: {(selectedObject.bounds.x.max - selectedObject.bounds.x.min).toFixed(2)}m</li>
                        <li>Depth: {(selectedObject.bounds.y.max - selectedObject.bounds.y.min).toFixed(2)}m</li>
                        <li>Height: {(selectedObject.bounds.z.max - selectedObject.bounds.z.min).toFixed(2)}m</li>
                      </ul>
                      <div>Position:</div>
                      <ul>
                        <li>X: {selectedObject.bounds.x.min.toFixed(2)}-{selectedObject.bounds.x.max.toFixed(2)}</li>
                        <li>Y: {selectedObject.bounds.y.min.toFixed(2)}-{selectedObject.bounds.y.max.toFixed(2)}</li>
                        <li>Z: {selectedObject.bounds.z.min.toFixed(2)}-{selectedObject.bounds.z.max.toFixed(2)}</li>
                      </ul>
                    </div>
                  </div>
                  <button className="btn btn-sm btn-outline-success w-100">
                    Highlight in 3D View
                  </button>
                </Card.Body>
              </Card>
            )}
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;