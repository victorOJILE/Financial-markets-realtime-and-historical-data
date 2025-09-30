//+------------------------------------------------------------------+
//|                                              CustomTrendFinder.mqh |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property strict

// A simple structure to hold swing point data
struct SwingPoint {
 long time;
 double price;
};

// A structure to store the best trend line information
struct TrendInfo {
 double r_squared;
 double slope;
 double intercept;
 long points_time[];
};

class CustomTrendFinder {
 private: double m_min_r_squared;
 int m_min_points;
 int m_num_trees;
 int m_residual_check_points;
 double m_min_residual_check_pct;
 
 // Internal helper functions
 bool calculateLinearRegression(const SwingPoint & points[], double & r_squared, double & slope, double & intercept);
 bool checkResiduals(const TrendInfo & trend,
  const SwingPoint & all_points[],
   const SwingPoint & trend_points[]);
 
 public:
  // Constructor
  void CustomTrendFinder(double min_r_squared = 0.99, int min_points = 3, int num_trees = 100, int residual_check_points = 10, double min_residual_check_pct = 0.8);
 
 // Main method to find the trend
 bool FindTrend(const SwingPoint & swing_points[], TrendInfo & trend_info);
};
//+------------------------------------------------------------------+
//| Class Constructor                                                |
//+------------------------------------------------------------------+
void CustomTrendFinder::CustomTrendFinder(double min_r_squared = 0.99, int min_points = 3, int num_trees = 100, int residual_check_points = 10, double min_residual_check_pct = 0.8) {
 m_min_r_squared = min_r_squared;
 m_min_points = min_points;
 m_num_trees = num_trees;
 m_residual_check_points = residual_check_points;
 m_min_residual_check_pct = min_residual_check_pct;
}

//+------------------------------------------------------------------+
//| Main method to find the trend                                    |
//+------------------------------------------------------------------+
bool CustomTrendFinder::FindTrend(const SwingPoint & swing_points[], TrendInfo & trend_info) {
 int total_points = ArraySize(swing_points);
 if (total_points < m_min_points) return false;
 
 TrendInfo best_trend_candidate;
 bool found_valid_candidate = false;
 
 for (int i = 0; i < m_num_trees; i++) {
  // 1. Random Subset Selection
  int num_points_to_select = (int) MathRand() % (total_points - m_min_points + 1) + m_min_points;
  SwingPoint selected_points[];
  ArrayResize(selected_points, num_points_to_select);
  
  int all_indices[];
  ArrayResize(all_indices, total_points);
  for (int k = 0; k < total_points; k++) all_indices[k] = k;
  
  // Shuffle indices
  for (int k = 0; k < total_points; k++) {
   int rnd_index = (int) MathRand() % total_points;
   int temp = all_indices[k];
   all_indices[k] = all_indices[rnd_index];
   all_indices[rnd_index] = temp;
  }
  
  // Select the first 'num_points_to_select' indices
  for (int k = 0; k < num_points_to_select; k++) selected_points[k] = swing_points[all_indices[k]];
  
  // Sort selected points by time to ensure order
  ArraySort(selected_points, 0, WHOLE_ARRAY, SORT_BY_VALUE, TIME_MODE);
  
  double r_squared, slope, intercept;
  if (calculateLinearRegression(selected_points, r_squared, slope, intercept)) {
   if (r_squared >= m_min_r_squared) {
    TrendInfo current_candidate;
    current_candidate.r_squared = r_squared;
    current_candidate.slope = slope;
    current_candidate.intercept = intercept;
    
    if (checkResiduals(current_candidate, swing_points, selected_points)) {
     // A validated candidate is found
     if (!found_valid_candidate || current_candidate.r_squared > best_trend_candidate.r_squared) {
      best_trend_candidate = current_candidate;
      found_valid_candidate = true;
      
      ArrayResize(best_trend_candidate.points_time, num_points_to_select);
      for (int k = 0; k < num_points_to_select; k++) {
       best_trend_candidate.points_time[k] = selected_points[k].time;
      }
     }
    }
   }
  }
 }
 
 if (found_valid_candidate) {
  trend_info = best_trend_candidate;
  return true;
 }
 
 return false;
}

//+------------------------------------------------------------------+
//| Calculates linear regression (slope, intercept, r-squared)       |
//+------------------------------------------------------------------+
bool CustomTrendFinder::calculateLinearRegression(const SwingPoint & points[], double & r_squared, double & slope, double & intercept) {
 int n = ArraySize(points);
 if (n < 2) return false;
 
 double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
 for (int i = 0; i < n; i++) {
  sum_x += points[i].time;
  sum_y += points[i].price;
  sum_xy += points[i].time * points[i].price;
  sum_x2 += points[i].time * points[i].time;
 }
 
 double denominator = n * sum_x2 - sum_x * sum_x;
 if (MathAbs(denominator) < DBL_EPSILON) {
  slope = 0;
  intercept = sum_y / n;
  r_squared = 0;
  return true; // Horizontal line
 }
 
 slope = (n * sum_xy - sum_x * sum_y) / denominator;
 intercept = (sum_y - slope * sum_x) / n;
 
 double y_mean = sum_y / n;
 double ss_total = 0, ss_residual = 0;
 for (int i = 0; i < n; i++) {
  double predicted_y = slope * points[i].time + intercept;
  ss_total += MathPow(points[i].price - y_mean, 2);
  ss_residual += MathPow(points[i].price - predicted_y, 2);
 }
 
 if (MathAbs(ss_total) < DBL_EPSILON) r_squared = 1.0;
 else r_squared = 1.0 - (ss_residual / ss_total);
 
 return true;
}

//+------------------------------------------------------------------+
//| Performs the residual check on earlier swing points              |
//+------------------------------------------------------------------+
bool CustomTrendFinder::checkResiduals(const TrendInfo & trend,
 const SwingPoint & all_points[],
  const SwingPoint & trend_points[]) {
 int total_points = ArraySize(all_points);
 int trend_points_size = ArraySize(trend_points);
 
 // Find the earliest point's index in the full array
 long earliest_trend_time = trend_points[0].time;
 int earliest_index = -1;
 for (int i = 0; i < total_points; i++) {
  if (all_points[i].time == earliest_trend_time) {
   earliest_index = i;
   break;
  }
 }
 
 if (earliest_index <= 0) return true; // No earlier points to check
 
 int start_index = MathMax(0, earliest_index - m_residual_check_points);
 int valid_residuals_count = 0;
 int points_checked_count = 0;
 
 bool is_uptrend = trend.slope > 0;
 
 for (int i = start_index; i < earliest_index; i++) {
  points_checked_count++;
  double predicted_y = trend.slope * all_points[i].time + trend.intercept;
  double residual = all_points[i].price - predicted_y;
  
  if ((is_uptrend && residual > 0) || (!is_uptrend && residual < 0)) valid_residuals_count++;
 }
 
 if (points_checked_count == 0) return true; // No points were checked
 
 return ((double) valid_residuals_count / points_checked_count) >= m_min_residual_check_pct;
}