import { Outlet } from "react-router-dom"
import { Sidebar } from "./Sidebar"
import { AssistantWidget } from "../assistant/AssistantWidget"

export function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <Outlet />
      </main>
      <AssistantWidget />
    </div>
  )
}
