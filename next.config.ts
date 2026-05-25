import type { NextConfig } from "next";

const BOT_API_URL = process.env.BOT_API_URL || "http://127.0.0.1:8100";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      // Bot API proxy - all /api/bot/* routes go to FastAPI
      {
        source: "/api/bot/:path*",
        destination: `${BOT_API_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
