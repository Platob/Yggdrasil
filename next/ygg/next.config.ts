import type { NextConfig } from "next";

const botPort = process.env.YGG_BOT_PORT || "8100";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `http://127.0.0.1:${botPort}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
