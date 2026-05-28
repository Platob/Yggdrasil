import type { NextConfig } from "next";

const BOT_API_URL = process.env.BOT_API_URL || "http://127.0.0.1:8100";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      // Proxy /api/v2/* to the FastAPI backend
      {
        source: "/api/v2/:path*",
        destination: `${BOT_API_URL}/api/v2/:path*`,
      },
      {
        source: "/api/card",
        destination: `${BOT_API_URL}/api/card`,
      },
      {
        source: "/api/ping",
        destination: `${BOT_API_URL}/api/ping`,
      },
    ];
  },
};

export default nextConfig;
