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

/** Training split / pipeline options (see TrainingLab handleCreate). */
export interface SplitConfig {
  test_after?: string;
  val_pct?: number;
  version_id?: number;
  include_builtin_features?: boolean;
  include_sp_features?: boolean;
  include_pace_shape_features?: boolean;
  include_race_relative_features?: boolean;
  include_elo_features?: boolean;
  include_odds_snapshot_features?: boolean;
  include_h2h_features?: boolean;
  optuna_objective?: string;
  walk_forward_folds?: number;
  embargo_days?: number;
  apply_monotone_constraints?: boolean;
  [key: string]: unknown;
}

/** Reliability-curve bins from ml/evaluation.compute_calibration_data. */
export interface CalibrationCurve {
  predicted_prob: number[];
  actual_freq: number[];
  bin_counts: number[];
}

export interface PnlPoint {
  race: number;
  pnl: number;
  fav_pnl?: number;
}

/** Betting simulation block from ml/evaluation (test-set $1 bets + Kelly). */
export interface BettingSimulation {
  top_pick_pnl: number;
  top_pick_races: number;
  top_pick_roi: number;
  top_pick_strike_rate: number;
  top_pick_winners: number;
  value_bet_pnl: number;
  value_bet_count: number;
  value_bet_roi: number;
  kelly_pnl?: number;
  kelly_races?: number;
  kelly_roi?: number;
  favourite_pnl: number;
  favourite_roi: number;
  pnl_by_race?: PnlPoint[];
  kelly_pnl_by_race?: { race: number; pnl: number }[];
}

export interface Experiment {
  id: number;
  name: string;
  description: string | null;
  algorithm: string;
  target: string;
  hyperparameters: Record<string, unknown>;
  feature_set: number[];
  split_config: SplitConfig | null;
  status: string;
  metrics: Record<string, number> | null;
  confusion_matrix: number[][] | null;
  calibration_data: {
    calibration?: CalibrationCurve | null;
    betting?: BettingSimulation | null;
  } | null;
  roc_data: { fpr: number[]; tpr: number[] } | null;
  shap_summary: unknown;
  feature_importance: Record<string, number> | null;
  training_duration_s: number | null;
  error_message: string | null;
  training_log: string | null;
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
  place_probability: number | null;
  show_probability: number | null;
  predicted_position: number | null;
  predicted_time: number | null;
  confidence: number | null;
  dog_name: string | null;
  trap: number | null;
}

export interface ForecastCombo {
  first_entry_id: number;
  second_entry_id: number;
  probability: number;
  first_dog?: string | null;
  first_trap?: number | null;
  second_dog?: string | null;
  second_trap?: number | null;
}

export interface TrioCombo {
  first_entry_id: number;
  second_entry_id: number;
  third_entry_id: number;
  probability: number;
  first_dog?: string | null;
  first_trap?: number | null;
  second_dog?: string | null;
  second_trap?: number | null;
  third_dog?: string | null;
  third_trap?: number | null;
}

export interface RaceCombosResponse {
  race_id: number;
  race_date: string;
  race_number: number | null;
  track_name: string | null;
  distance_m: number | null;
  grade: string | null;
  experiment_id: number;
  place_show: Array<{
    race_entry_id: number;
    dog_name: string | null;
    trap: number | null;
    place_probability: number | null;
    show_probability: number | null;
  }>;
  forecast_combos: ForecastCombo[];
  trio_combos: TrioCombo[];
}
