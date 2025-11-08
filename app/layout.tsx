import "./global.css";
import type { Metadata } from "next";
import React from "react";

export const metadata: Metadata = {
  title: "VLab Chatbot",
  description: "Virtual Lab Chatbot powered by Next.js",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        suppressHydrationWarning
        className="min-h-screen bg-gradient-to-b from-indigo-50 via-white to-blue-50 text-gray-900 antialiased"
      >
        {children}
      </body>
    </html>
  );
}
