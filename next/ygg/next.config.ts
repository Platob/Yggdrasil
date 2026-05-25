import type { NextConfig } from "next";

const NODE_API_URL = process.env.NODE_API_URL || `http://127.0.0.1:${process.env.YGG_NODE_PORT || "8100"}`;

const nextConfig: NextConfig = {
  turbopack: {
    root: process.cwd(),
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
