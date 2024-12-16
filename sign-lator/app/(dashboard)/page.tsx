"use client";
import React, { useEffect, useState } from "react";
import VideoFeed from "@/components/VideoFeed";

function Home() {
  const [message, setMessage] = useState("Loading...");

  useEffect(() => {
    fetch("http://localhost:8000/api/home")
      .then((response) => response.json())
      .then((data) => {
        setMessage(data.message);
      });
  }, []);

  return (
    <div>
      <h1>Webcam Video Capture</h1>
      <div>{message}</div>
      <VideoFeed />
    </div>
  );
}

export default Home;
