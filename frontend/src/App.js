import React, { useState } from "react";
import "./App.css"; // Your custom styles

function App() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [elaImage, setElaImage] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState("dark-theme"); // Toggle theme

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setImagePreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setElaImage(null);
      setHeatmap(null);
      setError(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to analyze image.");
      }

      const data = await response.json();
      setResult(data.result);
      setElaImage(data.ela_image || null);
      setHeatmap(data.heatmap || null);
    } catch (error) {
      setError("Error analyzing image.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const toggleTheme = () => {
    setTheme(theme === "dark-theme" ? "light-theme" : "dark-theme");
  };

  return (
    <div className={`app-container ${theme}`}>
      {/* Sidebar */}
      <div className="sidebar">
        <h2>Forgery Detector</h2>
        <ul>
          <li>Upload Image</li>
          <li>Analyze</li>
          <li>Results</li>
        </ul>
        <button className="theme-toggle" onClick={toggleTheme}>
          {theme === "dark-theme" ? "🌞 Light Mode" : "🌙 Dark Mode"}
        </button>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <h1>🔍 Image Forgery Detection</h1>

        {/* Upload Section */}
        <form onSubmit={handleUpload} className="upload-section">
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <button type="submit" className="upload-btn">
            Upload & Analyze
          </button>
        </form>

        {/* Image Preview */}
        {imagePreview && (
          <div className="image-container">
            <h3>Uploaded Image</h3>
            <img src={imagePreview} alt="Uploaded Preview" className="image-preview" />
          </div>
        )}

        {loading && <p>⏳ Analyzing image...</p>}
        {error && <p style={{ color: "red" }}>❌ {error}</p>}

        {/* Result Display */}
        {result && (
          <div className="result-card">
            <h2>
              Result:{" "}
              <span style={{ color: result === "Forged" ? "red" : "green" }}>
                {result}
              </span>
            </h2>

            {/* ELA Image */}
            {elaImage && (
              <div className="heatmap-section">
                <h3>ELA Image</h3>
                <img
                  src={`data:image/jpeg;base64,${elaImage}`}
                  alt="ELA"
                  className="large-image"
                />
              </div>
            )}

            {/* Heatmap */}
            {heatmap && (
              <div className="heatmap-section">
                <h3>Forgery Heatmap</h3>
                <img
                  src={`data:image/jpeg;base64,${heatmap}`}
                  alt="Heatmap"
                  className="large-image"
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
