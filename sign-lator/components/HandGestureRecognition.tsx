// filepath: my-nextjs-app/components/HandGestureRecognition.tsx
"use client"
import React, { useRef, useEffect, useState } from "react";

const HandGestureRecognition: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [landmarks, setLandmarks] = useState(null);

  useEffect(() => {
    const video = videoRef.current;

    if (video) {
      const captureImage = async () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
        const image = canvas.toDataURL("image/png");

        const response = await fetch("http://localhost:5000/detect", {
          method: "POST",
          body: JSON.stringify({ image }),
          headers: {
            "Content-Type": "application/json",
          },
        });
        const data = await response.json();
        setLandmarks(data.landmarks);
      };

      const interval = setInterval(captureImage, 1000);
      return () => clearInterval(interval);
    }
  }, []);

  return (
    <div>
      <video ref={videoRef} autoPlay style={{ display: "none" }} />
      <canvas width={640} height={480} />
      {landmarks && <div>Landmarks: {JSON.stringify(landmarks)}</div>}
    </div>
  );
};

export default HandGestureRecognition;
