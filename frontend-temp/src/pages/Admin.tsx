import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { apiService } from '@/services/api';
import { 
  AlertTriangle, 
  Database, 
  Trash2, 
  RefreshCw, 
  BarChart3,
  FileText,
  MessageSquare,
  HardDrive,
  Zap,
  Shield,
  AlertCircle,
  CheckCircle
} from 'lucide-react';

interface DatabaseStats {
  documents_count: number;
  document_chunks_count: number;
  chat_sessions_count: number;
  chat_messages_count: number;
  policies_count: number;
  bill_analyses_count: number;
  uploaded_files_count: number;
  uploaded_files_size_mb: number;
  embedding_dimensions: Record<string, number>;
}

const Admin: React.FC = () => {
  const [stats, setStats] = useState<DatabaseStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [resetLoading, setResetLoading] = useState(false);
  const [cleanupLoading, setCleanupLoading] = useState(false);
  const [clearStorageLoading, setClearStorageLoading] = useState(false);
  const [environment, setEnvironment] = useState<string>('development');
  const { toast } = useToast();

  // Load database stats on component mount
  useEffect(() => {
    loadDatabaseStats();
  }, []);

  const loadDatabaseStats = async () => {
    setLoading(true);
    try {
      const response = await apiService.getDatabaseStats();
      setStats(response.database_stats);
      setEnvironment(response.environment);
    } catch (error) {
      console.error('Failed to load database stats:', error);
      toast({
        title: "Failed to Load Stats",
        description: error instanceof Error ? error.message : "Could not fetch database statistics",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleResetDatabase = async () => {
    // Double confirmation for dangerous operation
    const firstConfirm = window.confirm(
      "⚠️ DANGER: This will permanently delete ALL data including documents, chats, and uploaded files. Are you sure?"
    );
    
    if (!firstConfirm) return;

    const secondConfirm = window.confirm(
      "🔥 FINAL WARNING: This action cannot be undone. Type 'YES' to proceed."
    );
    
    if (!secondConfirm) return;

    setResetLoading(true);
    try {
      const response = await apiService.resetAllData();
      
      // Clear frontend localStorage as well
      localStorage.removeItem('userProfile');
      localStorage.removeItem('insuranceData');
      localStorage.removeItem('currentSessionId');

      toast({
        title: "Database Reset Complete! 🔥",
        description: `Deleted ${response.reset_details.documents_deleted} documents, ${response.reset_details.chunks_deleted} chunks, and ${response.reset_details.files_deleted} files. Frontend storage also cleared.`,
      });

      // Reload stats to show empty database
      await loadDatabaseStats();
      
      // Reload the page to reset frontend state
      setTimeout(() => {
        window.location.href = '/';
      }, 2000);
      
    } catch (error) {
      console.error('Database reset failed:', error);
      toast({
        title: "Reset Failed",
        description: error instanceof Error ? error.message : "Failed to reset database",
        variant: "destructive",
      });
    } finally {
      setResetLoading(false);
    }
  };

  const handleCleanupEmbeddings = async () => {
    setCleanupLoading(true);
    try {
      const response = await apiService.cleanupEmbeddings();
      
      if (response.cleanup_result.mismatched_chunks_removed > 0) {
        toast({
          title: "Cleanup Complete! 🧹",
          description: `Removed ${response.cleanup_result.mismatched_chunks_removed} chunks with mismatched embeddings.`,
        });
      } else {
        toast({
          title: "No Cleanup Needed",
          description: "All embedding dimensions are already consistent.",
        });
      }

      // Reload stats
      await loadDatabaseStats();
      
    } catch (error) {
      console.error('Embedding cleanup failed:', error);
      toast({
        title: "Cleanup Failed",
        description: error instanceof Error ? error.message : "Failed to cleanup embeddings",
        variant: "destructive",
      });
    } finally {
      setCleanupLoading(false);
    }
  };

  const handleClearFrontendStorage = async () => {
    const confirm = window.confirm(
      "Clear all frontend storage (localStorage)? This will reset the current session."
    );
    
    if (!confirm) return;

    setClearStorageLoading(true);
    try {
      // Clear all localStorage
      localStorage.removeItem('userProfile');
      localStorage.removeItem('insuranceData');
      localStorage.removeItem('currentSessionId');
      
      toast({
        title: "Frontend Storage Cleared! 🧹",
        description: "All localStorage data has been cleared. Redirecting to home page...",
      });

      // Redirect to home to reset state
      setTimeout(() => {
        window.location.href = '/';
      }, 1500);
      
    } catch (error) {
      toast({
        title: "Clear Failed",
        description: error instanceof Error ? error.message : "Failed to clear frontend storage",
        variant: "destructive",
      });
    } finally {
      setClearStorageLoading(false);
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Shield className="h-8 w-8 text-red-600" />
          <h1 className="text-3xl font-bold">Admin Dashboard</h1>
          <Badge variant={environment === 'production' ? 'destructive' : 'secondary'}>
            {environment.toUpperCase()}
          </Badge>
        </div>
        <p className="text-muted-foreground">
          Database management and system administration tools
        </p>
      </div>

      {/* Environment Warning */}
      {environment === 'production' && (
        <Card className="mb-6 border-red-200 bg-red-50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-red-700">
              <AlertTriangle className="h-5 w-5" />
              <span className="font-semibold">Production Environment Detected</span>
            </div>
            <p className="text-red-600 mt-1">
              Database reset operations are disabled in production for safety.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4 text-blue-600" />
              Documents
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.documents_count || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats?.uploaded_files_count || 0} files ({formatBytes((stats?.uploaded_files_size_mb || 0) * 1024 * 1024)})
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Database className="h-4 w-4 text-green-600" />
              Chunks
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.document_chunks_count || 0}</div>
            <p className="text-xs text-muted-foreground">
              Text chunks with embeddings
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-purple-600" />
              Chat Sessions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.chat_sessions_count || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats?.chat_messages_count || 0} total messages
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-orange-600" />
              Analyses
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(stats?.policies_count || 0) + (stats?.bill_analyses_count || 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats?.policies_count || 0} policies, {stats?.bill_analyses_count || 0} bills
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Embedding Dimensions */}
      {stats?.embedding_dimensions && Object.keys(stats.embedding_dimensions).length > 0 && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-600" />
              Embedding Dimensions
            </CardTitle>
            <CardDescription>
              Distribution of embedding vector dimensions in the database
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(stats.embedding_dimensions).map(([dimension, count]) => (
                <div key={dimension} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {dimension.includes('768') ? (
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    ) : dimension.includes('384') ? (
                      <AlertCircle className="h-4 w-4 text-yellow-600" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-red-600" />
                    )}
                    <span className="font-medium">{dimension}</span>
                  </div>
                  <Badge variant="outline">{count} chunks</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Admin Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Database Operations */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Database className="h-5 w-5 text-blue-600" />
              Database Operations
            </CardTitle>
            <CardDescription>
              Manage database content and system state
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button 
              onClick={loadDatabaseStats}
              disabled={loading}
              variant="outline"
              className="w-full flex items-center gap-2"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh Statistics
            </Button>

            <Button 
              onClick={handleCleanupEmbeddings}
              disabled={cleanupLoading || loading}
              variant="outline"
              className="w-full flex items-center gap-2"
            >
              <Zap className={`h-4 w-4 ${cleanupLoading ? 'animate-pulse' : ''}`} />
              {cleanupLoading ? 'Cleaning...' : 'Cleanup Embeddings'}
            </Button>

            <Button 
              onClick={handleClearFrontendStorage}
              disabled={clearStorageLoading || loading}
              variant="outline"
              className="w-full flex items-center gap-2"
            >
              <HardDrive className={`h-4 w-4 ${clearStorageLoading ? 'animate-pulse' : ''}`} />
              {clearStorageLoading ? 'Clearing...' : 'Clear Frontend Storage'}
            </Button>

            <Separator />

            <div className="bg-red-50 p-4 rounded-lg border border-red-200">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="h-5 w-5 text-red-600" />
                <span className="font-semibold text-red-800">Danger Zone</span>
              </div>
              <p className="text-red-700 text-sm mb-3">
                This will permanently delete all documents, chats, analyses, and uploaded files.
              </p>
              <Button 
                onClick={handleResetDatabase}
                disabled={resetLoading || loading || environment === 'production'}
                variant="destructive"
                className="w-full flex items-center gap-2"
              >
                <Trash2 className={`h-4 w-4 ${resetLoading ? 'animate-pulse' : ''}`} />
                {resetLoading ? 'Resetting...' : 'Reset All Data'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* System Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <HardDrive className="h-5 w-5 text-green-600" />
              System Information
            </CardTitle>
            <CardDescription>
              Current system status and environment details
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Environment:</span>
                <Badge variant={environment === 'production' ? 'destructive' : 'secondary'}>
                  {environment}
                </Badge>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Total Storage:</span>
                <span className="text-sm">{formatBytes((stats?.uploaded_files_size_mb || 0) * 1024 * 1024)}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Database Status:</span>
                <Badge variant="outline" className="text-green-600 border-green-300">
                  Connected
                </Badge>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Last Updated:</span>
                <span className="text-sm text-muted-foreground">
                  {new Date().toLocaleTimeString()}
                </span>
              </div>
            </div>

            <Separator />

            <div className="text-xs text-muted-foreground">
              <p className="mb-1">⚠️ Admin operations require development environment</p>
              <p>💾 Always backup important data before reset operations</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Admin;
