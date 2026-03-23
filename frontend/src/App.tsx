import { Navigate, Route, Routes } from "react-router-dom";
import TopNav from "./components/TopNav";
import ConfigurePage from "./pages/ConfigurePage";
import HistoryPage from "./pages/HistoryPage";
import ResultsPage from "./pages/ResultsPage";
import UploadPage from "./pages/UploadPage";

export default function App() {
  return (
    <div className="min-h-screen">
      <TopNav />
      <main>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/configure" element={<ConfigurePage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
