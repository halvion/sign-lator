"use client";
import React, { useEffect, useState } from "react";

const VideoFeed: React.FC = () => {
    const [data, setData] = useState(null);

    useEffect(() => {
        fetch("http://localhost:8000/api/home")
            .then((response) => response.json())
            .then((data) => setData(data))
            .catch((error) => console.error("Error fetching data:", error));
    }, []);

    return (
        <div>
            <h2>Video Feed</h2>
            <img src="http://localhost:8000/api/video_feed" width="960" height="540" />
        </div>
    );
};

export default VideoFeed;