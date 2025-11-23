import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { apiService, PolicyAnalysisResponse, Document } from '@/services/api';

interface UserProfile {
  insuranceUploaded: boolean;
  insuranceFile?: string;
  insuranceAnalysis?: PolicyAnalysisResponse;
  documentId?: number;
}

interface AppContextType {
  // User state
  userProfile: UserProfile | null;
  setUserProfile: (profile: UserProfile | null) => void;
  
  // Insurance state
  insuranceData: PolicyAnalysisResponse | null;
  setInsuranceData: (data: PolicyAnalysisResponse | null) => void;
  
  // Chat state
  currentSessionId: string | null;
  setCurrentSessionId: (sessionId: string | null) => void;
  
  // Documents state
  documents: Document[];
  setDocuments: (documents: Document[]) => void;
  
  // Loading states
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  
  // Error state
  error: string | null;
  setError: (error: string | null) => void;
  
  // Backend health
  backendHealthy: boolean;
  setBackendHealthy: (healthy: boolean) => void;
  healthCheckLoading: boolean;
  
  // Helper functions
  uploadInsurance: (file: File) => Promise<PolicyAnalysisResponse>;
  uploadBill: (file: File) => Promise<{ bill_id: string; filename: string }>;
  createChatSession: () => Promise<string>;
  checkBackendHealth: () => Promise<boolean>;
  loadUserProfile: () => void;
  clearUserData: () => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const useApp = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  // State
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [insuranceData, setInsuranceData] = useState<PolicyAnalysisResponse | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [healthCheckLoading, setHealthCheckLoading] = useState(true);

  // Load user profile from localStorage on mount
  const loadUserProfile = () => {
    try {
      const stored = localStorage.getItem('userProfile');
      if (stored) {
        const profile = JSON.parse(stored);
        setUserProfile(profile);
        
        // Load insurance data if available
        const insuranceStored = localStorage.getItem('insuranceData');
        if (insuranceStored) {
          setInsuranceData(JSON.parse(insuranceStored));
        }
      }
    } catch (error) {
      console.error('Error loading user profile:', error);
      setError('Failed to load user profile');
    }
  };

  // Clear all user data
  const clearUserData = () => {
    setUserProfile(null);
    setInsuranceData(null);
    setCurrentSessionId(null);
    setDocuments([]);
    localStorage.removeItem('userProfile');
    localStorage.removeItem('insuranceData');
    localStorage.removeItem('currentSessionId');
  };

  // Check backend health
  const checkBackendHealth = async (): Promise<boolean> => {
    try {
      const health = await apiService.checkHealth();
      const healthy = health.status === 'healthy' || health.status === 'degraded';
      setBackendHealthy(healthy);
      setHealthCheckLoading(false);
      return healthy;
    } catch (error) {
      console.error('Backend health check failed:', error);
      setBackendHealthy(false);
      setHealthCheckLoading(false);
      return false;
    }
  };

  // Upload insurance document
  const uploadInsurance = async (file: File): Promise<PolicyAnalysisResponse> => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Upload and analyze insurance document
      const analysisResult = await apiService.uploadInsurance(file);
      
      // Update state
      setInsuranceData(analysisResult);
      
      const updatedProfile: UserProfile = {
        insuranceUploaded: true,
        insuranceFile: file.name,
        insuranceAnalysis: analysisResult,
      };
      
      setUserProfile(updatedProfile);
      
      // Persist to localStorage
      localStorage.setItem('userProfile', JSON.stringify(updatedProfile));
      localStorage.setItem('insuranceData', JSON.stringify(analysisResult));
      
      // Refresh documents list
      try {
        const docsResponse = await apiService.getDocuments();
        setDocuments(docsResponse.documents);
      } catch (docError) {
        console.warn('Failed to refresh documents:', docError);
      }
      
      return analysisResult;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload insurance';
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  // Upload medical bill
  const uploadBill = async (file: File): Promise<{ bill_id: string; filename: string }> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const uploadResult = await apiService.uploadBill(file);
      
      // Refresh documents list
      try {
        const docsResponse = await apiService.getDocuments();
        setDocuments(docsResponse.documents);
      } catch (docError) {
        console.warn('Failed to refresh documents:', docError);
      }
      
      return {
        bill_id: uploadResult.bill_id,
        filename: uploadResult.filename,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload bill';
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  // Create chat session
  const createChatSession = async (): Promise<string> => {
    try {
      // Get available document IDs if user has uploaded documents
      // Use null instead of undefined to match working frontend
      const documentIds = documents.length > 0 ? documents.map(doc => doc.id) : null;
      
      const sessionResponse = await apiService.createChatSession(documentIds);
      const sessionId = sessionResponse.session_id;
      
      setCurrentSessionId(sessionId);
      localStorage.setItem('currentSessionId', sessionId);
      
      return sessionId;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create chat session';
      setError(errorMessage);
      throw error;
    }
  };

  // Load documents on mount and when user profile changes
  useEffect(() => {
    if (userProfile?.insuranceUploaded) {
      apiService.getDocuments()
        .then(response => setDocuments(response.documents))
        .catch(error => console.warn('Failed to load documents:', error));
    }
  }, [userProfile]);

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
    
    // Check health periodically
    const healthInterval = setInterval(checkBackendHealth, 30000); // Every 30 seconds
    
    return () => clearInterval(healthInterval);
  }, []);

  // Load user profile on mount
  useEffect(() => {
    loadUserProfile();
  }, []);

  const value: AppContextType = {
    userProfile,
    setUserProfile,
    insuranceData,
    setInsuranceData,
    currentSessionId,
    setCurrentSessionId,
    documents,
    setDocuments,
    isLoading,
    setIsLoading,
    error,
    setError,
    backendHealthy,
    setBackendHealthy,
    healthCheckLoading,
    uploadInsurance,
    uploadBill,
    createChatSession,
    checkBackendHealth,
    loadUserProfile,
    clearUserData,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

export default AppProvider;
