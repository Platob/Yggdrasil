import type { NextConfig } from "next";

const BOT_API_URL = process.env.BOT_API_URL || `http://127.0.0.1:${process.env.YGG_BOT_PORT || "8100"}`;

const nextConfig: NextConfig = {
  turbopack: {
    root: ".",
  },
  async rewrites() {
    return [
      {
        source: "/api/bot/:path*",
        destination: `${BOT_API_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
