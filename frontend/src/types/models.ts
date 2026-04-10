export interface Track {
  id: number;
  name: string;
  code: string;
  location: string | null;
  distances_m: number[] | null;
  surface: string;
  num_traps: number;
  active: boolean;
}

export interface Dog {
  id: number;
  name: string;
  sire: string | null;
  dam: string | null;
  birth_date: string | null;
  sex: string | null;
  colour: string | null;
  trainer_name: string | null;
  owner_name: string | null;
  greyhound_data_id: string | null;
  gri_id: string | null;
}

export interface RaceEntry {
  id: number;
  race_id: number;
  dog_id: number;
  trap: number;
  finish_position: number | null;
  finish_time: number | null;
  sectional_time: number | null;
  beaten_distance: number | null;
  weight_kg: number | null;
  starting_price: string | null;
  sp_decimal: number | null;
  comment: string | null;
  dog_name: string | null;
}

export interface Race {
  id: number;
  track_id: number;
  race_date: string;
  race_time: string | null;
  race_number: number | null;
  distance_m: number;
  grade: string | null;
  race_type: string;
  prize_money: number | null;
  going: string | null;
  num_runners: number | null;
  status: string;
  track_name: string | null;
}

export interface RaceDetail extends Race {
  entries: RaceEntry[];
}

export interface FeatureDefinition {
  id: number;
  name: string;
  display_name: string | null;
  description: string | null;
  feature_type: 'visual' | 'code';
  config_json: Record<string, unknown> | null;
  code: string | null;
  input_columns: string[] | null;
  output_dtype: string;
  enabled: boolean;
}

export interface Experiment {
  id: number;
  name: string;
  description: string | null;
  algorithm: string;
  target: string;
  hyperparameters: Record<string, unknown>;
  feature_set: number[];
  split_config: Record<string, unknown> | null;
  status: string;
  metrics: Record<string, number> | null;
  confusion_matrix: unknown;
  calibration_data: unknown;
  roc_data: unknown;
  shap_summary: unknown;
  feature_importance: Record<string, number> | null;
  training_duration_s: number | null;
  error_message: string | null;
  created_at: string | null;
  completed_at: string | null;
  heartbeat_at: string | null;
  training_stage: string | null;
}

export interface Prediction {
  id: number;
  experiment_id: number;
  race_entry_id: number;
  win_probability: number | null;
  predicted_position: number | null;
  predicted_time: number | null;
  confidence: number | null;
  dog_name: string | null;
  trap: number | null;
}
