export async function GET() {
  return Response.json({
    version: "1.0.0",
    features: {
      chat: true,
      execute: true,
      remoteCall: true,
    },
  });
}
