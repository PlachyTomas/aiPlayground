import { Link, Route, BrowserRouter as Router, Routes } from "react-router-dom";
import Dashboard from "./routes/Dashboard";
import Datasets from "./routes/Datasets";
import Labeling from "./routes/Labeling";
import Models from "./routes/Models";
import Train from "./routes/Train";

export default function App() {
  return (
    <Router>
      <nav style={{ display: "flex", gap: 12 }}>
        <Link to="/">Dashboard</Link>
        <Link to="/datasets">Datasets</Link>
        <Link to="/labeling">Labeling</Link>
        <Link to="/train">Train</Link>
        <Link to="/models">Models</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/labeling" element={<Labeling />} />
        <Route path="/train" element={<Train />} />
        <Route path="/models" element={<Models />} />
      </Routes>
    </Router>
  );
}
