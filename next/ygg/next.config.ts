import type { NextConfig } from "next";

const NODE_API_URL = process.env.NODE_API_URL || `http://127.0.0.1:${process.env.YGG_NODE_PORT || "8100"}`;

const nextConfig: NextConfig = {
  turbopack: {
    root: process.cwd(),
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Access-Control-Allow-Methods", value: "GET,POST,PUT,DELETE,PATCH,OPTIONS" },
          { key: "Access-Control-Allow-Headers", value: "*" },
          { key: "Access-Control-Allow-Credentials", value: "true" },
        ],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/api/node/:path*",
        destination: `${NODE_API_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
