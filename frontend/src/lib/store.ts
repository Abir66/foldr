import { create } from "zustand";

export type SheetSnap = "peek" | "half" | "full";
export type Speed = 0.5 | 1 | 2;

interface UiState {
  /** current step, 0 = flat sheet, n = finished */
  step: number;
  /** continuous playback position k + t (step k at progress t), reported by the 3D viewer */
  pos: number;
  /** the fold being shown (0-based step index; pos - foldK = its progress 0..1) */
  foldK: number;
  playing: boolean;
  speed: Speed;
  sheet: SheetSnap;
  libraryOpen: boolean;
  aboutOpen: boolean;
  /** Fold mode: which side of the paper faces up at the start */
  foldSide: "white" | "blue";
  setFoldSide: (s: "white" | "blue") => void;
  setStep: (k: number) => void;
  setPlaying: (p: boolean) => void;
  setSpeed: (s: Speed) => void;
  setSheet: (s: SheetSnap) => void;
  setLibraryOpen: (o: boolean) => void;
  setAboutOpen: (o: boolean) => void;
}

export const useUi = create<UiState>((set) => ({
  step: 0,
  pos: 0,
  foldK: 0,
  playing: false,
  speed: 1,
  sheet: "half",
  libraryOpen: false,
  aboutOpen: false,
  foldSide: "white",
  setFoldSide: (foldSide) => set({ foldSide }),
  setStep: (step) => set({ step, pos: step, foldK: Math.max(0, step - 1) }),
  setPlaying: (playing) => set({ playing }),
  setSpeed: (speed) => set({ speed }),
  setSheet: (sheet) => set({ sheet }),
  setLibraryOpen: (libraryOpen) => set({ libraryOpen }),
  setAboutOpen: (aboutOpen) => set({ aboutOpen }),
}));
