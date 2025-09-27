import React, { useState, useEffect, useRef } from "react";
import {
  CssBaseline,
  ThemeProvider,
  createTheme,
  Box,
  Typography,
  Button,
  TextField,
  Paper,
  IconButton,
  Chip,
  Avatar,
  Divider,
  Tooltip,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  AppBar,
  Toolbar,
  Card,
  CardContent,
  CardActions,
  Modal,
  Snackbar,
  Alert,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  LinearProgress,
} from "@mui/material";
import {
  CloudUpload,
  Logout,
  Search,
  InsertDriveFile,
  Summarize,
  Mood,
  Tag,
  Help,
  Image,
  Delete,
  Description,
  Tune,
  Shield,
  Home,
  Settings,
  Menu,
  Download,
  CheckCircle,
  Error,
  Info,
} from "@mui/icons-material";
import axios from "axios";

const SIDEBAR_WIDTH = 220;
const theme = createTheme({
  palette: { mode: "dark", primary: { main: "#6366f1" }, background: { default: "#17181c" } },
  typography: { fontFamily: "Inter, Montserrat, Arial, sans-serif" },
  components: {
    MuiCard: { styleOverrides: { root: { borderRadius: 18, boxShadow: "0 6px 32px rgba(68,72,212,0.09)" } } },
    MuiPaper: { styleOverrides: { root: { borderRadius: 18 } } },
    MuiChip: { styleOverrides: { root: { fontWeight: 500, fontSize: "1rem" } } },
    MuiButton: { styleOverrides: { root: { textTransform: "none", borderRadius: 12, fontWeight: 600 } } },
    MuiDrawer: { styleOverrides: { paper: { width: SIDEBAR_WIDTH, boxSizing: "border-box" } } },
  },
});

const api = axios.create({ baseURL: "http://localhost:5000" });
api.interceptors.request.use(config => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

const AuthContext = React.createContext();
const useAuth = () => React.useContext(AuthContext);

function AuthProvider({ children }) {
  const [auth, setAuth] = useState(() => {
    const token = localStorage.getItem("token");
    const user = JSON.parse(localStorage.getItem("user") || "null");
    return token && user ? { token, user } : null;
  });
  const login = async (email, password) => {
    try {
      const res = await api.post("/login", { email, password });
      localStorage.setItem("token", res.data.access_token);
      localStorage.setItem("user", JSON.stringify(res.data.user));
      setAuth({ token: res.data.access_token, user: res.data.user });
      return { success: true };
    } catch (e) {
      return { success: false, error: e.response?.data?.error || "Login failed" };
    }
  };
  const register = async (email, password, name) => {
    try {
      const res = await api.post("/register", { email, password, name });
      localStorage.setItem("token", res.data.access_token);
      localStorage.setItem("user", JSON.stringify(res.data.user));
      setAuth({ token: res.data.access_token, user: res.data.user });
      return { success: true };
    } catch (e) {
      return { success: false, error: e.response?.data?.error || "Registration failed" };
    }
  };
  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setAuth(null);
  };
  return (
    <AuthContext.Provider value={{ auth, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

function Login({ onLogin, onSwitch }) {
  const [email, setEmail] = useState(""); 
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false); 
  const [error, setError] = useState("");
  const handleSubmit = async e => {
    e.preventDefault(); setLoading(true); setError("");
    const res = await onLogin(email, password); setLoading(false);
    if (!res.success) setError(res.error);
  };
  return (
    <Box sx={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "linear-gradient(120deg, #232536 0%, #252543 100%)" }}>
      <Paper sx={{ p: 6, maxWidth: 400, width: "100%", background: "#22223a", borderRadius: 4 }}>
        <Typography variant="h3" fontWeight={900} sx={{ textAlign: "center", mb: 4, fontFamily: "Inter", color: "#A78BFA" }}>Nasuni AI</Typography>
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Email"
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            sx={{
              mb: 3,
              '& .MuiInputBase-root': { background: "#232536", color: "#fff" },
              '& .MuiInputLabel-root': { color: "#A78BFA" },
              '& .MuiOutlinedInput-notchedOutline': { borderColor: "#6366f1" }
            }}
            InputLabelProps={{ style: { color: "#A78BFA" } }}
            InputProps={{ style: { color: "#fff" } }}
            required
          />
          <TextField
            fullWidth
            label="Password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            sx={{
              mb: 4,
              '& .MuiInputBase-root': { background: "#232536", color: "#fff" },
              '& .MuiInputLabel-root': { color: "#A78BFA" },
              '& .MuiOutlinedInput-notchedOutline': { borderColor: "#6366f1" }
            }}
            InputLabelProps={{ style: { color: "#A78BFA" } }}
            InputProps={{ style: { color: "#fff" } }}
            required
          />
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          <Button type="submit" fullWidth variant="contained" disabled={loading} sx={{ py: 1.5, mb: 2 }}>{loading ? "Signing in..." : "Sign In"}</Button>
          <Button fullWidth variant="outlined" onClick={onSwitch} sx={{ py: 1.5 }}>Create New Account</Button>
        </form>
      </Paper>
    </Box>
  );
}

function Register({ onRegister, onSwitch }) {
  const [email, setEmail] = useState(""); 
  const [password, setPassword] = useState(""); 
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false); 
  const [error, setError] = useState("");
  const handleSubmit = async e => {
    e.preventDefault(); setLoading(true); setError("");
    const res = await onRegister(email, password, name); setLoading(false);
    if (!res.success) setError(res.error);
  };
  return (
    <Box sx={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "linear-gradient(120deg, #232536 0%, #252543 100%)" }}>
      <Paper sx={{ p: 6, maxWidth: 400, width: "100%", background: "#22223a", borderRadius: 4 }}>
        <Typography variant="h3" fontWeight={900} sx={{ textAlign: "center", mb: 4, fontFamily: "Inter", color: "#A78BFA" }}>Create Account</Typography>
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Full Name"
            value={name}
            onChange={e => setName(e.target.value)}
            sx={{
              mb: 3,
              '& .MuiInputBase-root': { background: "#232536", color: "#fff" },
              '& .MuiInputLabel-root': { color: "#A78BFA" },
              '& .MuiOutlinedInput-notchedOutline': { borderColor: "#6366f1" }
            }}
            InputLabelProps={{ style: { color: "#A78BFA" } }}
            InputProps={{ style: { color: "#fff" } }}
            required
          />
          <TextField
            fullWidth
            label="Email"
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            sx={{
              mb: 3,
              '& .MuiInputBase-root': { background: "#232536", color: "#fff" },
              '& .MuiInputLabel-root': { color: "#A78BFA" },
              '& .MuiOutlinedInput-notchedOutline': { borderColor: "#6366f1" }
            }}
            InputLabelProps={{ style: { color: "#A78BFA" } }}
            InputProps={{ style: { color: "#fff" } }}
            required
          />
          <TextField
            fullWidth
            label="Password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            sx={{
              mb: 4,
              '& .MuiInputBase-root': { background: "#232536", color: "#fff" },
              '& .MuiInputLabel-root': { color: "#A78BFA" },
              '& .MuiOutlinedInput-notchedOutline': { borderColor: "#6366f1" }
            }}
            InputLabelProps={{ style: { color: "#A78BFA" } }}
            InputProps={{ style: { color: "#fff" } }}
            required
          />
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          <Button type="submit" fullWidth variant="contained" disabled={loading} sx={{ py: 1.5, mb: 2 }}>{loading ? "Creating..." : "Create Account"}</Button>
          <Button fullWidth variant="outlined" onClick={onSwitch} sx={{ py: 1.5 }}>Back to Login</Button>
        </form>
      </Paper>
    </Box>
  );
}

function AnalysisTab({ endpoint, title, inputLabel, placeholder, isFileUpload = false, allowFileSelect = false }) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);
  const [selectedFileId, setSelectedFileId] = useState("");
  const [files, setFiles] = useState([]);
  const { auth } = useAuth();

  // Load user files list
  useEffect(() => {
    if (allowFileSelect && auth) {
      const loadFiles = async () => {
        try {
          const res = await api.get("/files");
          setFiles(res.data.files || []);
        } catch (e) {
          console.error("Failed to load files:", e);
        }
      };
      loadFiles();
    }
  }, [allowFileSelect, auth]);

  const handleAnalyze = async () => {
    setResult(null);
    setLoading(true);
    try {
      if (isFileUpload) {
        if (!file) { 
          setResult({ error: "Please select a file to analyze." }); 
          setLoading(false); 
          return; 
        }
        const formData = new FormData();
        formData.append("file", file);
        const res = await api.post(endpoint, formData, { 
          headers: { "Content-Type": "multipart/form-data" } 
        });
        setResult(res.data);
      } else if (allowFileSelect && selectedFileId) {
        // Use file ID for analysis
        let analysisEndpoint = endpoint;
        let analysisType = "";
        
        // Determine analysis type
        if (endpoint === "/api/analyze/summary") {
          analysisType = "summary";
        } else if (endpoint === "/api/analyze/sentiment") {
          analysisType = "sentiment";
        } else if (endpoint === "/api/analyze/keywords") {
          analysisType = "keywords";
        } else if (endpoint === "/api/analyze/ner") {
          analysisType = "ner";
        } else if (endpoint === "/api/analyze/qa") {
          analysisType = "qa";
          analysisEndpoint = `/api/analyze/file/${selectedFileId}/qa?question=${encodeURIComponent(input)}`;
        } else if (endpoint === "/api/analyze/semantic") {
          analysisType = "semantic";
          analysisEndpoint = `/api/analyze/file/${selectedFileId}/semantic?query=${encodeURIComponent(input)}`;
        } else if (endpoint === "/api/analyze/toxicity") {
          analysisType = "toxicity";
        }
        
        if (analysisType && !analysisEndpoint.includes("/api/analyze/file/")) {
          analysisEndpoint = `/api/analyze/file/${selectedFileId}/${analysisType}`;
        }
        
        const res = await api.get(analysisEndpoint);
        setResult(res.data);
      } else {
        if (!input.trim()) { 
          setResult({ error: "Please enter text." }); 
          setLoading(false); 
          return; 
        }
        const payload = (endpoint === "/api/analyze/qa")
          ? { question: input, context: "" }
          : { text: input };
        const res = await api.post(endpoint, payload);
        setResult(res.data);
      }
    } catch (e) {
      setResult({ error: e.response?.data?.error || "Request failed" });
    }
    setLoading(false);
  };

  // Format results for better display
  const renderResult = () => {
    if (!result) return null;
    
    if (result.error) {
      return (
        <Alert severity="error" sx={{ mt: 2 }}>
          {result.error}
        </Alert>
      );
    }
    
    // Format different analysis types
    if (endpoint === "/api/analyze/summary") {
      return (
        <Paper sx={{ p: 3, mt: 3, background: "#232536" }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
            Summary:
          </Typography>
          <Typography sx={{ lineHeight: 1.6 }}>
            {result.summary || "No summary generated"}
          </Typography>
          {result.model && (
            <Typography variant="caption" sx={{ color: "#aaa", mt: 1, display: "block" }}>
              Model: {result.model}
            </Typography>
          )}
          {result.key_points_covered && (
            <Typography variant="caption" sx={{ color: "#aaa", mt: 1, display: "block" }}>
              Key points covered: {result.key_points_covered}
            </Typography>
          )}
        </Paper>
      );
    }
    
    if (endpoint === "/api/analyze/keywords") {
      return (
        <Paper sx={{ p: 3, mt: 3, background: "#232536" }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
            Keywords:
          </Typography>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
            {result.keywords?.map((kw, idx) => (
              <Chip 
                key={idx} 
                label={kw} 
                color="primary" 
                variant="filled"
                sx={{ fontWeight: 500 }}
              />
            ))}
          </Box>
          {result.method && (
            <Typography variant="caption" sx={{ color: "#aaa", mt: 1, display: "block" }}>
              Method: {result.method}
            </Typography>
          )}
        </Paper>
      );
    }
    
    if (endpoint === "/api/analyze/sentiment") {
      const getSentimentColor = (sentiment) => {
        if (sentiment === "positive") return "success";
        if (sentiment === "negative") return "error";
        return "info";
      };
      
      return (
        <Paper sx={{ p: 3, mt: 3, background: "#232536" }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
            Sentiment Analysis:
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <Chip 
              label={result.sentiment || "neutral"} 
              color={getSentimentColor(result.sentiment)}
              sx={{ fontWeight: 600, fontSize: "1rem" }}
            />
            <Typography>
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </Typography>
          </Box>
          {result.scores && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Detailed Scores:</Typography>
              {result.scores.map((score, idx) => (
                <Box key={idx} sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                  <Typography>{score.label}:</Typography>
                  <Typography>{(score.score * 100).toFixed(1)}%</Typography>
                </Box>
              ))}
            </Box>
          )}
        </Paper>
      );
    }
    
    if (endpoint === "/api/analyze/ner") {
      return (
        <Paper sx={{ p: 3, mt: 3, background: "#232536" }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
            Named Entities:
          </Typography>
          {result.entities?.length > 0 ? (
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
              {result.entities.map((entity, idx) => (
                <Chip 
                  key={idx} 
                  label={`${entity.text} (${entity.type})`}
                  variant="outlined"
                  sx={{ borderColor: "#6366f1", color: "#fff" }}
                />
              ))}
            </Box>
          ) : (
            <Typography>No entities found</Typography>
          )}
          {result.model && (
            <Typography variant="caption" sx={{ color: "#aaa", mt: 1, display: "block" }}>
              Model: {result.model}
            </Typography>
          )}
        </Paper>
      );
    }
    
    if (endpoint === "/api/analyze/qa") {
      return (
        <Paper sx={{ p: 3, mt: 3, background: "#232536" }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
            Answer:
          </Typography>
          <Typography sx={{ fontSize: "1.1rem", lineHeight: 1.6 }}>
            {result.qa_result?.answer || "No answer found"}
          </Typography>
          {result.qa_result?.score !== undefined && (
            <Typography variant="caption" sx={{ color: "#aaa", mt: 1, display: "block" }}>
              Confidence: {(result.qa_result.score * 100).toFixed(1)}%
            </Typography>
          )}
        </Paper>
      );
    }
    
    if (endpoint === "/api/analyze/semantic") {
      return (
        <Paper sx={{ p: 3, mt: 3, background: "#232536" }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
            Semantic Similarity Results:
          </Typography>
          {Array.isArray(result) && result.length > 0 ? (
            <Box>
              {result.map((item, idx) => (
                <Box key={idx} sx={{ mb: 2, p: 2, background: "#1a1a2e", borderRadius: 1 }}>
                  <Typography sx={{ mb: 1 }}>
                    {item.document?.substring(0, 150)}
                    {item.document?.length > 150 ? "..." : ""}
                  </Typography>
                  <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
                    <Chip 
                      label={`Similarity: ${(item.similarity * 100).toFixed(1)}%`}
                      color="primary"
                      size="small"
                    />
                  </Box>
                </Box>
              ))}
            </Box>
          ) : (
            <Typography>No similar documents found</Typography>
          )}
        </Paper>
      );
    }
    
    if (endpoint === "/api/analyze/toxicity") {
      return (
        <Paper sx={{ p: 3, mt: 3, background: "#232536" }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
            Toxicity Analysis:
          </Typography>
          {result.toxicity_scores ? (
            <Box>
              {Object.entries(result.toxicity_scores).map(([label, score]) => (
                <Box key={label} sx={{ mb: 1.5 }}>
                  <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                    <Typography>{label}:</Typography>
                    <Typography>{(score * 100).toFixed(1)}%</Typography>
                  </Box>
                  <Box sx={{ width: "100%", height: 8, background: "#333", borderRadius: 4, overflow: "hidden" }}>
                    <Box 
                      sx={{ 
                        width: `${score * 100}%`, 
                        height: "100%", 
                        background: score > 0.7 ? "#f44336" : score > 0.4 ? "#ff9800" : "#4caf50",
                        borderRadius: 4
                      }} 
                    />
                  </Box>
                </Box>
              ))}
              {result.severity && (
                <Box sx={{ mt: 2, p: 1, background: "#1a1a2e", borderRadius: 1 }}>
                  <Typography>Severity: {result.severity}</Typography>
                </Box>
              )}
            </Box>
          ) : (
            <Typography>No toxicity data available</Typography>
          )}
        </Paper>
      );
    }
    
    if (endpoint === "/api/analyze/ocr") {
      return (
        <Paper sx={{ p: 3, mt: 3, background: "#232536" }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
            OCR Result:
          </Typography>
          {result.ocr_text ? (
            <Typography sx={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
              {result.ocr_text}
            </Typography>
          ) : result.error ? (
            <Alert severity="error">{result.error}</Alert>
          ) : (
            <Typography>No text extracted</Typography>
          )}
          {result.method && (
            <Typography variant="caption" sx={{ color: "#aaa", mt: 1, display: "block" }}>
              Method: {result.method}
            </Typography>
          )}
        </Paper>
      );
    }
    
    // Default display for other results
    return (
      <Paper sx={{ p: 3, mt: 3, background: "#232536" }}>
        <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
          Result:
        </Typography>
        <pre style={{ whiteSpace: "pre-wrap", fontSize: "0.9rem" }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      </Paper>
    );
  };

  return (
    <Paper sx={{ p: 4, borderRadius: 4, background: "#18181b" }}>
      <Typography variant="h5" fontWeight={700} sx={{ mb: 3, fontFamily: "Inter" }}>
        {title}
      </Typography>

      {allowFileSelect && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            Select a file to analyze:
          </Typography>
          <FormControl fullWidth>
            <InputLabel id="file-select-label">File</InputLabel>
            <Select
              labelId="file-select-label"
              value={selectedFileId}
              label="File"
              onChange={(e) => setSelectedFileId(e.target.value)}
              sx={{ mb: 2 }}
            >
              <MenuItem value="">
                <em>None</em>
              </MenuItem>
              {files.map((file) => (
                <MenuItem key={file.file_id} value={file.file_id}>
                  {file.filename}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      )}

      {isFileUpload ? (
        <>
          <Button component="label" variant="outlined" startIcon={<CloudUpload />} sx={{ mb: 2 }}>
            Choose File
            <input type="file" hidden onChange={e => setFile(e.target.files?.[0] || null)} />
          </Button>
          <Typography sx={{ mb: 2 }}>{file ? `Selected: ${file.name}` : "No file selected"}</Typography>
        </>
      ) : (
        <TextField
          fullWidth
          multiline
          minRows={4}
          label={inputLabel}
          placeholder={placeholder}
          value={input}
          onChange={e => setInput(e.target.value)}
          sx={{ mb: 3 }}
          disabled={!!selectedFileId && (endpoint !== "/api/analyze/qa" && endpoint !== "/api/analyze/semantic")}
        />
      )}

      <Button variant="contained" onClick={handleAnalyze} disabled={loading} sx={{ mr: 2 }}>
        {loading ? "Analyzing..." : "Run Analysis"}
      </Button>

      {loading && <LinearProgress sx={{ mt: 2 }} />}
      {renderResult()}
    </Paper>
  );
}

function Dashboard() {
  const { auth, logout } = useAuth();
  const [profile, setProfile] = useState(null);
  const [files, setFiles] = useState([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [tab, setTab] = useState(0);
  const [fileModal, setFileModal] = useState({ open: false, file: null });
  const [fileToUpload, setFileToUpload] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [snackbar, setSnackbar] = useState({ open: false, message: "", severity: "success" });
  const fileInputRef = useRef();

  const [isMobile, setIsMobile] = useState(window.innerWidth < 900);
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 900);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => { loadProfile(); loadFiles(); }, []);

  const showSnackbar = (message, severity = "success") => setSnackbar({ open: true, message, severity });
  const loadProfile = async () => {
    try { const res = await api.get("/profile"); setProfile(res.data.user); }
    catch (e) { showSnackbar("Failed to load profile", "error"); }
  };
  const loadFiles = async () => {
    try { const res = await api.get("/files"); setFiles(res.data.files || []); }
    catch (e) { showSnackbar("Failed to load files", "error"); }
  };

  const handleFileInputChange = e => { setFileToUpload(e.target.files[0]); };
  const handleFileUpload = async e => {
    e.preventDefault();
    if (!fileToUpload) return showSnackbar("Please select a file first", "warning");
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", fileToUpload);
      await api.post("/upload", formData, { headers: { "Content-Type": "multipart/form-data" } });
      setFileToUpload(null);
      fileInputRef.current.value = "";
      showSnackbar("File uploaded and processed successfully!");
      await loadFiles();
    } catch (e) { showSnackbar("Upload failed: " + (e.response?.data?.error || "Unknown error"), "error"); }
    setUploading(false);
  };
  
  const handleDeleteFile = async (fileId) => {
    if (!fileId) {
      console.error("No fileId provided!");
      return showSnackbar("Delete failed: fileId missing", "error");
    }
    if (!window.confirm("Are you sure you want to delete this file?")) return;
    try {
      await api.delete(`/files/${fileId}`);
      showSnackbar("File deleted successfully!");
      await loadFiles();
    } catch (e) {
      console.error("Delete error:", e.response?.data || e.message);
      showSnackbar("Failed to delete file", "error");
    }
  };
  
  const handleSearch = async () => {
    if (!searchQuery.trim()) return await loadFiles();
    try {
      const res = await api.post("/search", { query: searchQuery });
      setFiles(res.data.results.map(r => r.file));
      showSnackbar(`Found ${res.data.results.length} results`);
    } catch (e) { showSnackbar("Search failed", "error"); }
  };
  
  const openFileModal = file => setFileModal({ open: true, file });
  const closeFileModal = () => setFileModal({ open: false, file: null });

  // Download Feature
  const handleDownloadFile = async (file) => {
    try {
      const response = await api.get(`/files/${file.file_id}/download`, { responseType: 'blob' });
      const filename = file.filename || "downloaded_file";
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      showSnackbar("Download failed", "error");
    }
  };

  // Health check
  const [healthInfo, setHealthInfo] = useState(null);
  const loadHealth = async () => {
    try {
      const res = await api.get("/health");
      setHealthInfo(res.data);
    } catch (e) {
      setHealthInfo({ error: "Health check failed" });
    }
  };

  const sidebarItems = [
    { icon: <Home />, text: "Dashboard", tab: 0 },
    { icon: <Summarize />, text: "Summarization", tab: 1, endpoint: "/api/analyze/summary" },
    { icon: <Mood />, text: "Sentiment", tab: 2, endpoint: "/api/analyze/sentiment" },
    { icon: <Tag />, text: "Keywords", tab: 3, endpoint: "/api/analyze/keywords" },
    // { icon: <Shield />, text: "Toxicity", tab: 4, endpoint: "/api/analyze/toxicity" },
    { icon: <Help />, text: "NER", tab: 5, endpoint: "/api/analyze/ner" },
    { icon: <Description />, text: "QA", tab: 6, endpoint: "/api/analyze/qa" },
    { icon: <Tune />, text: "Semantic", tab: 7, endpoint: "/api/analyze/semantic" },
    { icon: <Image />, text: "OCR/Image", tab: 8, endpoint: "/api/analyze/ocr" },
    // { icon: <Settings />, text: "Health", tab: 9, endpoint: "/health" },
  ];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: "flex", minHeight: "100vh", background: "linear-gradient(120deg,#232536 0%,#252543 100%)" }}>
        <Drawer
          variant={isMobile ? "temporary" : "permanent"}
          open={drawerOpen || !isMobile}
          onClose={() => setDrawerOpen(false)}
          PaperProps={{
            sx: {
              background: "linear-gradient(135deg,#232536 80%,#6366f1 100%)",
              color: "#fff",
              width: SIDEBAR_WIDTH,
              boxShadow: "0 10px 32px rgba(68,72,212,0.09)",
              borderRight: 0,
            },
          }}
        >
          <Box sx={{ py: 4, height: "100%", display: "flex", flexDirection: "column" }}>
            <Typography variant="h4" align="center" fontWeight={900} sx={{ letterSpacing: 1, mb: 6, fontFamily: "Inter" }}>
              Nasuni AI
            </Typography>
            <List sx={{ flex: 1 }}>
              {sidebarItems.map((item) => (
                <ListItemButton
                  key={item.text}
                  selected={tab === item.tab}
                  onClick={() => {
                    setTab(item.tab);
                    if (item.tab === 9) loadHealth();
                    if (isMobile) setDrawerOpen(false);
                  }}
                  sx={{
                    borderRadius: 2,
                    mb: 1,
                    py: 1.5,
                    px: 2,
                    bgcolor: tab === item.tab ? "rgba(99,102,241,0.1)" : "transparent",
                    "&:hover": { bgcolor: "rgba(99,102,241,0.05)" }
                  }}
                >
                  <ListItemIcon sx={{ color: "#818cf8", minWidth: 32 }}>{item.icon}</ListItemIcon>
                  <ListItemText
                    primary={item.text}
                    primaryTypographyProps={{
                      fontWeight: tab === item.tab ? 700 : 500,
                      fontFamily: "Inter",
                      fontSize: "1.08rem",
                    }}
                  />
                </ListItemButton>
              ))}
            </List>
            <Button
              variant="contained"
              color="secondary"
              endIcon={<Logout />}
              onClick={logout}
              sx={{
                mx: 2,
                mb: 2,
                width: "calc(100% - 32px)",
                bgcolor: "#A78BFA",
                fontWeight: 600,
                fontFamily: "Inter",
                borderRadius: 2,
                fontSize: "1rem",
                "&:hover": { bgcolor: "#8B5CF6" }
              }}
            >
              LOGOUT
            </Button>
          </Box>
        </Drawer>

        <Box
          sx={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            ...(isMobile ? {} : { ml: `${SIDEBAR_WIDTH}px` }),
            transition: "margin-left 0.2s",
          }}
        >
          <AppBar
            position="static"
            elevation={0}
            sx={{
              bgcolor: "#111114",
              borderBottom: "1px solid #2d2f3b",
              px: 2,
              py: 1,
            }}
          >
            <Toolbar sx={{ display: "flex", alignItems: "center", minHeight: 72 }}>
              {isMobile && (
                <IconButton onClick={() => setDrawerOpen(true)} sx={{ color: "#fff", mr: 2 }}>
                  <Menu />
                </IconButton>
              )}
              <Box sx={{ width: isMobile ? 0 : SIDEBAR_WIDTH, flexShrink: 0 }} />
              <Box sx={{ display: "flex", alignItems: "center", flex: 1, px: 2, gap: 2 }}>
                <TextField
                  variant="outlined"
                  placeholder="Search summaries, keywords..."
                  size="small"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleSearch()}
                  sx={{
                    bgcolor: "#232536",
                    borderRadius: 2,
                    fontFamily: "Inter",
                    color: "#fff",
                    flex: 1,
                    "& .MuiOutlinedInput-root": {
                      "& fieldset": { borderColor: "transparent" },
                      "&:hover fieldset": { borderColor: "#6366f1" },
                      "&.Mui-focused fieldset": { borderColor: "#6366f1" },
                    }
                  }}
                  InputProps={{
                    startAdornment: <Search sx={{ color: "#6366f1", mr: 1 }} />,
                  }}
                />
                <Button variant="contained" onClick={handleSearch} sx={{ bgcolor: "#6366f1", fontWeight: 600, borderRadius: 2, "&:hover": { bgcolor: "#4f46e5" } }}>
                  SEARCH
                </Button>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", ml: 4 }}>
                <Avatar
                  alt={profile?.name}
                  src={`https://api.dicebear.com/7.x/pixel-art/svg?seed=${profile?.name || "User"}`}
                  sx={{ width: 38, height: 38, mr: 1, border: "2px solid #6366f1" }}
                />
                <Typography fontWeight={700} sx={{ fontFamily: "Inter", fontSize: "1.1rem" }}>
                  {profile?.name}
                </Typography>
              </Box>
            </Toolbar>
          </AppBar>

          <Box sx={{ p: 4, flex: 1, overflow: "auto" }}>
            {/* TAB 0 — Main Dashboard & File Upload */}
            {tab === 0 && (
              <>
                <Paper elevation={4} sx={{ p: 4, mb: 4, borderRadius: 4, background: "#18181b" }}>
                  <Typography variant="h5" fontWeight={700} sx={{ mb: 3, fontFamily: "Inter" }}>
                    Upload & Analyze Files
                  </Typography>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 2, flexWrap: "wrap" }}>
                    <Button
                      component="label"
                      variant="outlined"
                      color="primary"
                      sx={{ borderRadius: 2, px: 3, py: 1.5, fontWeight: 600, fontFamily: "Inter", fontSize: "1.1rem", borderColor: "#6366f1", color: "#6366f1" }}
                      startIcon={<CloudUpload />}
                    >
                      Browse File
                      <input type="file" hidden ref={fileInputRef} onChange={handleFileInputChange} />
                    </Button>
                    {fileToUpload && (
                      <Typography fontWeight={500} sx={{ fontFamily: "Inter", fontSize: "1rem" }}>
                        Selected: {fileToUpload.name}
                      </Typography>
                    )}
                    <Button
                      variant="contained"
                      color="primary"
                      disabled={!fileToUpload || uploading}
                      onClick={handleFileUpload}
                      sx={{ borderRadius: 2, px: 3, py: 1.5, fontWeight: 600, fontFamily: "Inter", fontSize: "1.1rem", bgcolor: "#6366f1", "&:hover": { bgcolor: "#4f46e5" } }}
                    >
                      {uploading ? "Uploading..." : "Upload & Analyze"}
                    </Button>
                    <Button variant="outlined" onClick={() => { setFileToUpload(null); fileInputRef.current.value = ""; }} sx={{ ml: 2 }}>
                      Clear
                    </Button>
                  </Box>
                </Paper>

                <Typography variant="h5" fontWeight={900} sx={{ mb: 3, fontFamily: "Inter", color: "#fff" }}>
                  Your Files ({files.length})
                </Typography>

                {files.length === 0 ? (
                  <Paper sx={{ p: 6, textAlign: "center", background: "#18181b" }}>
                    <Typography variant="h6" sx={{ color: "#ccc", mb: 2 }}>No files uploaded yet</Typography>
                    <Typography sx={{ color: "#888" }}>Upload your first file to see it analyzed here</Typography>
                  </Paper>
                ) : (
                  <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", md: "1fr 1fr 1fr" }, gap: 3 }}>
                    {files.map(file => (
                      <Card key={file.file_id} sx={{ bgcolor: "#22223a", color: "#fff" }}>
                        <CardContent>
                          <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                            <InsertDriveFile sx={{ color: "#6366f1", mr: 1 }} />
                            <Typography
                              variant="subtitle1"
                              fontWeight={700}
                              sx={{
                                fontFamily: "Inter",
                                fontSize: "1.12rem",
                                whiteSpace: "nowrap",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                flex: 1,
                              }}
                              title={file.filename}
                            >
                              {file.filename}
                            </Typography>
                          </Box>
                          <Typography
                            sx={{
                              fontFamily: "Inter",
                              fontWeight: 500,
                              fontSize: "1.02rem",
                              mb: 2,
                              minHeight: 36,
                              display: "-webkit-box",
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: "vertical",
                              overflow: "hidden",
                            }}
                          >
                            {file.summary || "No summary available"}
                          </Typography>
                          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mb: 2 }}>
                            {file.keywords?.slice(0, 4).map((kw, idx) => (
                              <Chip
                                key={idx}
                                label={kw}
                                color="primary"
                                variant="filled"
                                size="small"
                                sx={{ fontFamily: "Inter", fontWeight: 600, fontSize: "0.95rem", bgcolor: "#6366f1", color: "#fff" }}
                              />
                            ))}
                            {file.keywords?.length > 4 && (
                              <Chip label={`+${file.keywords.length - 4}`} size="small" sx={{ fontFamily: "Inter" }} />
                            )}
                          </Box>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 1 }}>
                            <Chip
                              label={file.sentiment || "neutral"}
                              color={file.sentiment === "positive" ? "success" : file.sentiment === "negative" ? "error" : "primary"}
                              variant="outlined"
                              size="small"
                              sx={{ fontFamily: "Inter", fontWeight: 600, fontSize: "0.95rem" }}
                            />
                            <Typography variant="caption" sx={{ fontFamily: "Inter", color: "#bbb" }}>
                              {file.char_count} chars
                            </Typography>
                          </Box>
                        </CardContent>
                        <CardActions sx={{ justifyContent: "flex-end" }}>
                          <Tooltip title="Download file">
                            <IconButton color="primary" onClick={() => handleDownloadFile(file)}>
                              <Download />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="View details">
                            <IconButton color="primary" onClick={() => openFileModal(file)}>
                              <Description />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton color="error" onClick={() => handleDeleteFile(file.file_id)}>
                              <Delete />
                            </IconButton>
                          </Tooltip>
                        </CardActions>
                      </Card>
                    ))}
                  </Box>
                )}

                <Modal
                  open={fileModal.open}
                  onClose={closeFileModal}
                  sx={{ display: "flex", alignItems: "center", justifyContent: "center", p: 2 }}
                >
                  <Paper sx={{ p: 4, borderRadius: 4, background: "#232536", color: "#fff", maxWidth: 820, width: "100%", maxHeight: "80vh", overflow: "auto" }}>
                    {fileModal.file && (
                      <>
                        <Typography variant="h5" fontWeight={900} sx={{ mb: 2, fontFamily: "Inter" }}>
                          {fileModal.file.filename}
                        </Typography>
                        <Divider sx={{ mb: 3, bgcolor: "#6366f1" }} />
                        <Typography variant="subtitle1" fontWeight={600} sx={{ fontFamily: "Inter", mb: 1 }}>
                          Summary:
                        </Typography>
                        <Typography sx={{ mb: 3, fontFamily: "Inter", lineHeight: 1.6 }}>
                          {fileModal.file.summary || "No summary available"}
                        </Typography>
                        <Typography variant="subtitle1" fontWeight={600} sx={{ fontFamily: "Inter", mb: 1 }}>
                          Keywords:
                        </Typography>
                        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mb: 3 }}>
                          {fileModal.file.keywords?.map((kw, idx) => (
                            <Chip key={idx} label={kw} color="primary" variant="filled" size="small" sx={{ bgcolor: "#6366f1" }} />
                          ))}
                        </Box>

                        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" }, gap: 2, mb: 3 }}>
                          <Box>
                            <Typography variant="subtitle2" fontWeight={600} sx={{ fontFamily: "Inter" }}>
                              Sentiment:
                            </Typography>
                            <Chip
                              label={fileModal.file.sentiment || "neutral"}
                              color={fileModal.file.sentiment === "positive" ? "success" : fileModal.file.sentiment === "negative" ? "error" : "primary"}
                              size="small"
                            />
                          </Box>
                          <Box>
                            <Typography variant="subtitle2" fontWeight={600} sx={{ fontFamily: "Inter" }}>
                              Characters:
                            </Typography>
                            <Typography>{fileModal.file.char_count}</Typography>
                          </Box>
                        </Box>

                        {/* Quick AI actions for this file */}
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1 }}>Analyze this file:</Typography>
                          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                            <Button variant="outlined" onClick={async () => {
                              navigator.clipboard.writeText(fileModal.file.summary || "");
                              showSnackbar("File summary copied to clipboard", "success");
                            }}>Copy Summary</Button>
                            <Button variant="outlined" onClick={async () => {
                              try {
                                const payload = { question: "Give main points", context: fileModal.file.summary || "" };
                                const res = await api.post("/api/analyze/qa", payload);
                                showSnackbar("QA returned — see console", "success");
                                console.log("QA result", res.data);
                                alert(JSON.stringify(res.data, null, 2));
                              } catch (e) {
                                showSnackbar("QA failed", "error");
                              }
                            }}>Quick QA</Button>
                            <Button variant="outlined" onClick={async () => {
                              try {
                                const payload = { query: "summary", documents: [fileModal.file.summary || ""] };
                                const res = await api.post("/api/analyze/semantic", payload);
                                console.log("Semantic:", res.data);
                                alert(JSON.stringify(res.data, null, 2));
                              } catch (e) { showSnackbar("Semantic failed", "error"); }
                            }}>Semantic</Button>
                          </Box>
                        </Box>

                        <Button variant="contained" onClick={closeFileModal} fullWidth sx={{ bgcolor: "#6366f1", fontWeight: 700, "&:hover": { bgcolor: "#4f46e5" } }}>
                          Close
                        </Button>
                      </>
                    )}
                  </Paper>
                </Modal>
              </>
            )}

            {/* TAB 1..8 — Analysis Tabs */}
            {tab > 0 && tab < 9 && (
              <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 420px" }, gap: 3 }}>
                <Box>
                  {/* Map tab -> AnalysisTab component */}
                  {tab === 1 && <AnalysisTab endpoint="/api/analyze/summary" title="Summarization" inputLabel="Paste text to summarize" placeholder="Paste long text here..." allowFileSelect={true} />}
                  {tab === 2 && <AnalysisTab endpoint="/api/analyze/sentiment" title="Sentiment Analysis" inputLabel="Enter text to analyze sentiment" placeholder="Enter text..." allowFileSelect={true} />}
                  {tab === 3 && <AnalysisTab endpoint="/api/analyze/keywords" title="Keyword Extraction" inputLabel="Enter text to extract keywords" placeholder="Enter text..." allowFileSelect={true} />}
                  {/* {tab === 4 && <AnalysisTab endpoint="/api/analyze/toxicity" title="Toxicity Check" inputLabel="Enter text to check toxicity" placeholder="Enter text..." allowFileSelect={true} />} */}
                  {tab === 5 && <AnalysisTab endpoint="/api/analyze/ner" title="Named Entity Recognition (NER)" inputLabel="Enter text for NER" placeholder="Enter text..." allowFileSelect={true} />}
                  {tab === 6 && <AnalysisTab endpoint="/api/analyze/qa" title="Document Q&A" inputLabel="Enter question (context optional in backend)" placeholder="Ask a question..." allowFileSelect={true} />}
                  {tab === 7 && <AnalysisTab endpoint="/api/analyze/semantic" title="Semantic Search / Similarity" inputLabel="Enter query" placeholder="Enter search query..." allowFileSelect={true} />}
                  {tab === 8 && <AnalysisTab endpoint="/api/analyze/ocr" title="OCR / Image Analysis" inputLabel="" placeholder="" isFileUpload={true} />}
                </Box>

                <Box>
                  {/* Right side helper: quick file list + details */}
                  <Paper sx={{ p: 3, mb: 3, background: "#18181b" }}>
                    <Typography variant="h6" fontWeight={700} sx={{ mb: 2 }}>Quick Files</Typography>
                    <Box sx={{ maxHeight: 360, overflow: "auto" }}>
                      {files.map(f => (
                        <Box key={f.file_id} sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1, p: 1, borderRadius: 1, bgcolor: "#111" }}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                            <InsertDriveFile sx={{ color: "#6366f1" }} />
                            <Box>
                              <Typography sx={{ fontSize: 13, fontWeight: 700 }}>{f.filename}</Typography>
                              <Typography sx={{ fontSize: 11, color: "#bbb" }}>{f.char_count} chars</Typography>
                            </Box>
                          </Box>
                          <Box>
                            <Tooltip title="Use summary as input">
                              <Button size="small" onClick={() => {
                                navigator.clipboard.writeText(f.summary || "");
                                showSnackbar("File summary copied to clipboard", "success");
                              }}>Copy</Button>
                            </Tooltip>
                            <Tooltip title="Open file">
                              <IconButton size="small" onClick={() => openFileModal(f)}><Description /></IconButton>
                            </Tooltip>
                          </Box>
                        </Box>
                      ))}
                    </Box>
                  </Paper>

                  <Paper sx={{ p: 3, background: "#18181b" }}>
                    <Typography variant="h6" fontWeight={700} sx={{ mb: 2 }}>Help</Typography>
                    <Typography sx={{ fontSize: 13, color: "#ddd", mb: 2 }}>
                      Tip: Use the right panel to quickly copy a file's summary into the analysis input (copy button). For OCR, upload a PDF/image file.
                    </Typography>
                    <Button variant="outlined" onClick={() => showSnackbar("Docs: Use backend endpoints for advanced features", "info")}>Docs</Button>
                  </Paper>
                </Box>
              </Box>
            )}

            {/* TAB 9 — Health
            {tab === 9 && (
              <Paper sx={{ p: 4, borderRadius: 4, background: "#18181b" }}>
                <Typography variant="h5" fontWeight={700} sx={{ mb: 3 }}>Server Health</Typography>
                <Button variant="contained" onClick={loadHealth} sx={{ mb: 2 }}>Check Health</Button>
                {healthInfo ? (
                  <Paper sx={{ p: 2, background: "#232536" }}>
                    <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(healthInfo, null, 2)}</pre>
                  </Paper>
                ) : (
                  <Typography sx={{ color: "#bbb" }}>Click "Check Health" to ping the backend.</Typography>
                )}
              </Paper>
            )} */}
          </Box>
        </Box>
      </Box>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert severity={snackbar.severity} sx={{ fontFamily: "Inter" }}>{snackbar.message}</Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default function App() {
  const [screen, setScreen] = useState("login");
  return (
    <AuthProvider>
      <AuthContext.Consumer>
        {({ auth, login, register }) =>
          !auth ? (
            screen === "login" ? (
              <Login onLogin={login} onSwitch={() => setScreen("register")} />
            ) : (
              <Register onRegister={register} onSwitch={() => setScreen("login")} />
            )
          ) : (
            <Dashboard />
          )
        }
      </AuthContext.Consumer>
    </AuthProvider>
  );
}