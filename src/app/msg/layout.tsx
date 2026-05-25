import { ServiceLayout } from "@/components/service-layout";

export const metadata = {
  title: "Yggdrasil — Messaging",
  description: "Real-time messaging across Yggdrasil channels",
};

export default function MsgLayout({ children }: { children: React.ReactNode }) {
  return <ServiceLayout>{children}</ServiceLayout>;
}
