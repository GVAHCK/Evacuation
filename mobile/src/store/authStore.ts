import { create } from "zustand";
import * as SecureStore from "expo-secure-store";
import { User } from "../types";
import { Api, setAuthToken } from "../api/client";

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;
  loadToken: () => Promise<void>;
  login: (email: string, pwd: string) => Promise<boolean>;
  register: (name: string, email: string, pwd: string, role?: string) => Promise<boolean>;
  logout: () => Promise<void>;
  clearError: () => void;
}

const TOKEN_KEY = "evac_auth_token";

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isLoading: false,
  error: null,

  loadToken: async () => {
    set({ isLoading: true });
    try {
      let storedToken: string | null = null;
      try {
        storedToken = await SecureStore.getItemAsync(TOKEN_KEY);
      } catch (storageErr) {
        // SecureStore fallback if in web environment
        if (typeof window !== "undefined" && window.localStorage) {
          storedToken = window.localStorage.getItem(TOKEN_KEY);
        }
      }

      if (storedToken) {
        setAuthToken(storedToken);
        const user = await Api.getMe();
        set({ user, token: storedToken, isLoading: false });
      } else {
        setAuthToken(null);
        set({ user: null, token: null, isLoading: false });
      }
    } catch (err) {
      setAuthToken(null);
      set({ user: null, token: null, isLoading: false });
    }
  },

  login: async (email: string, pwd: string) => {
    set({ isLoading: true, error: null });
    try {
      const data = await Api.login(email, pwd);
      setAuthToken(data.access_token);
      try {
        await SecureStore.setItemAsync(TOKEN_KEY, data.access_token);
      } catch {
        if (typeof window !== "undefined" && window.localStorage) {
          window.localStorage.setItem(TOKEN_KEY, data.access_token);
        }
      }
      const user = await Api.getMe();
      set({ user, token: data.access_token, isLoading: false });
      return true;
    } catch (err: any) {
      const msg = err.response?.data?.message || "Authentication failed";
      set({ error: msg, isLoading: false });
      return false;
    }
  },

  register: async (name: string, email: string, pwd: string, role: string = "USER") => {
    set({ isLoading: true, error: null });
    try {
      await Api.register(name, email, pwd, role);
      // Auto-login after registration
      const data = await Api.login(email, pwd);
      setAuthToken(data.access_token);
      try {
        await SecureStore.setItemAsync(TOKEN_KEY, data.access_token);
      } catch {
        if (typeof window !== "undefined" && window.localStorage) {
          window.localStorage.setItem(TOKEN_KEY, data.access_token);
        }
      }
      const user = await Api.getMe();
      set({ user, token: data.access_token, isLoading: false });
      return true;
    } catch (err: any) {
      const msg = err.response?.data?.message || "Registration failed";
      set({ error: msg, isLoading: false });
      return false;
    }
  },

  logout: async () => {
    setAuthToken(null);
    try {
      await SecureStore.deleteItemAsync(TOKEN_KEY);
    } catch {
      if (typeof window !== "undefined" && window.localStorage) {
        window.localStorage.removeItem(TOKEN_KEY);
      }
    }
    set({ user: null, token: null });
  },

  clearError: () => set({ error: null }),
}));

