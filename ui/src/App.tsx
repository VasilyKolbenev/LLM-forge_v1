import { BrowserRouter, Routes, Route } from "react-router-dom"
import { Layout } from "@/components/layout/Layout"
import { Dashboard } from "@/pages/Dashboard"
import { NewExperiment } from "@/pages/NewExperiment"
import { Experiments } from "@/pages/Experiments"
import { Datasets } from "@/pages/Datasets"
import { Monitoring } from "@/pages/Monitoring"
import { Compute } from "@/pages/Compute"
import { WorkflowBuilder } from "@/pages/WorkflowBuilder"
import { PromptLab } from "@/pages/PromptLab"
import { Agent } from "@/pages/Agent"
import { Settings } from "@/pages/Settings"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/new" element={<NewExperiment />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/workflows" element={<WorkflowBuilder />} />
          <Route path="/monitoring" element={<Monitoring />} />
          <Route path="/compute" element={<Compute />} />
          <Route path="/prompts" element={<PromptLab />} />
          <Route path="/agent" element={<Agent />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
