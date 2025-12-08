import argparse
import json
import math
from typing import Dict

import pandas as pd
import requests
from geopy.distance import geodesic


class SolutionValidator:
    def __init__(
        self,
        distance_method: str = "haversine",
        cache_file: str = "distance_cache.json",
        matrix_file: str = None,
        solution_file: str = None,
    ):
        self.distance_method = distance_method
        self.cache_file = cache_file
        self.distance_cache = {}
        self.distance_matrix = None
        self.distance_matrix_file = matrix_file
        self.matrix_metadata = {}
        self.solution_file = solution_file or "verificacion_caso1.csv"

        # Load distance matrix if provided and method is matrix
        if matrix_file and distance_method == "matrix":
            self.load_distance_matrix(matrix_file)

        # Load cache only if not using matrix method
        if distance_method != "matrix":
            self.load_cache()

        try:
            # Load data
            self.vehicles_df = pd.read_csv("vehicles.csv")
            self.clients_df = pd.read_csv("clients.csv")
            self.depots_df = pd.read_csv("depots.csv")

            # Try configured solution file, fallback to solution.csv
            try:
                self.solution_df = pd.read_csv(self.solution_file)
            except FileNotFoundError:
                if self.solution_file != "solution.csv":
                    print(f"Warning: {self.solution_file} not found, trying solution.csv...")
                    self.solution_df = pd.read_csv("solution.csv")
                else:
                    raise

            # Create coordinate mappings
            self.locations = {}

            # Map depots with their alphanumeric identifiers
            # Based on your data, depot 1 maps to "CDA"
            for _, depot in self.depots_df.iterrows():
                depot_numeric_id = int(depot["DepotID"])

                # Map numeric ID
                self.locations[depot_numeric_id] = {
                    "latitude": depot["Latitude"],
                    "longitude": depot["Longitude"],
                    "type": "depot",
                }

                # Map alphanumeric ID (CDA for depot 1)
                if depot_numeric_id == 1:
                    self.locations["CDA"] = {
                        "latitude": depot["Latitude"],
                        "longitude": depot["Longitude"],
                        "type": "depot",
                    }

            # Add client coordinates
            for _, client in self.clients_df.iterrows():
                client_id = f"C{int(client['ClientID']):03d}"
                self.locations[client_id] = {
                    "latitude": client["Latitude"],
                    "longitude": client["Longitude"],
                    "type": "client",
                    "demand": int(client["Demand"]),
                }

            # Create lookup dictionaries
            self.client_demands = {
                f"C{int(row.ClientID):03d}": int(row.Demand)
                for _, row in self.clients_df.iterrows()
            }

            self.vehicle_specs = {
                int(row.VehicleID): {
                    "capacity": int(row.Capacity),
                    "range": int(row.Range),
                }
                for _, row in self.vehicles_df.iterrows()
            }

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def load_cache(self):
        """Load distance cache from file."""
        try:
            with open(self.cache_file, "r") as f:
                self.distance_cache = json.load(f)
        except FileNotFoundError:
            self.distance_cache = {}

    def save_cache(self):
        """Save distance cache to file."""
        with open(self.cache_file, "w") as f:
            json.dump(self.distance_cache, f)

    def load_distance_matrix(self, matrix_file: str) -> None:
        """
        Load distance matrix from JSON or CSV file.

        Args:
            matrix_file: Path to distance matrix file (.json or .csv)

        Raises:
            ValueError: If file format is invalid or contains errors
            FileNotFoundError: If file doesn't exist
        """
        import os

        if not os.path.exists(matrix_file):
            raise FileNotFoundError(
                f"Distance matrix file not found: {matrix_file}\n"
                "Ensure the file path is correct and the file exists."
            )

        # Detect format from extension
        file_ext = os.path.splitext(matrix_file)[1].lower()

        try:
            if file_ext == ".json":
                self._load_json_matrix(matrix_file)
            elif file_ext == ".csv":
                self._load_csv_matrix(matrix_file)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}\n"
                    "Distance matrix must be .json or .csv"
                )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON format in {matrix_file}:\n"
                f"  {str(e)}\n"
                "Ensure the file is valid JSON."
            )
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Invalid CSV format in {matrix_file}:\n"
                f"  {str(e)}\n"
                "Ensure the file is valid CSV."
            )

        print(f"Loaded distance matrix from {matrix_file}")
        if "matrix_type" in self.matrix_metadata:
            print(f"Matrix type: {self.matrix_metadata['matrix_type']}")

    def _load_json_matrix(self, matrix_file: str) -> None:
        """Load distance matrix from JSON file."""
        with open(matrix_file, "r") as f:
            data = json.load(f)

        # Extract metadata if present
        if "metadata" in data:
            self.matrix_metadata = data["metadata"]

        # Get distances
        if "distances" not in data:
            raise ValueError("JSON matrix must contain 'distances' key")

        distances = data["distances"]

        # Validate structure
        if not isinstance(distances, dict):
            raise ValueError("'distances' must be a dictionary")

        # Validate and store matrix
        self.distance_matrix = {}
        for origin, destinations in distances.items():
            if not isinstance(destinations, dict):
                raise ValueError(f"Destinations for {origin} must be a dictionary")

            self.distance_matrix[origin] = {}
            for dest, dist in destinations.items():
                if not isinstance(dist, (int, float)):
                    raise ValueError(
                        f"Distance {origin}→{dest} must be numeric, got {type(dist)}"
                    )
                if dist < 0:
                    raise ValueError(
                        f"Distance {origin}→{dest} cannot be negative: {dist}"
                    )
                if not math.isfinite(dist):
                    raise ValueError(
                        f"Distance {origin}→{dest} must be finite, got {dist}"
                    )

                self.distance_matrix[origin][dest] = dist

    def _load_csv_matrix(self, matrix_file: str) -> None:
        """Load distance matrix from CSV file."""
        df = pd.read_csv(matrix_file)

        # Check for three-column format (Origin, Destination, Distance)
        if "Origin" in df.columns and "Destination" in df.columns and "Distance" in df.columns:
            # Validate distance column
            if not pd.api.types.is_numeric_dtype(df["Distance"]):
                raise ValueError("Distance column must contain numeric values")

            if (df["Distance"] < 0).any():
                raise ValueError("Distance values cannot be negative")

            # Build matrix from three-column format
            self.distance_matrix = {}
            for _, row in df.iterrows():
                origin = str(row["Origin"])
                dest = str(row["Destination"])
                distance = float(row["Distance"])

                if origin not in self.distance_matrix:
                    self.distance_matrix[origin] = {}

                self.distance_matrix[origin][dest] = distance

        # Check for matrix format (first column is row labels)
        else:
            # First column is assumed to be row labels
            row_labels = df.iloc[:, 0].astype(str).tolist()
            col_labels = df.columns[1:].astype(str).tolist()

            # Build matrix
            self.distance_matrix = {}
            for i, origin in enumerate(row_labels):
                self.distance_matrix[origin] = {}
                for j, dest in enumerate(col_labels):
                    distance = float(df.iloc[i, j + 1])
                    if distance < 0:
                        raise ValueError(
                            f"Distance {origin}→{dest} cannot be negative: {distance}"
                        )
                    self.distance_matrix[origin][dest] = distance

    def normalize_location_id(self, loc_id: str) -> str:
        """
        Normalize location ID to standardized format.

        Handles conversions:
        - Numeric depot IDs (1, 2, 3) → Standardized (CD01, CD02, CD03)
        - Old depot IDs (CDA, CDB, CDC) → Standardized (CD01, CD02, CD03)
        - Numeric client IDs → Standardized (C001, C002, C003, ...)
        - Already standardized IDs → Return as-is

        Args:
            loc_id: Location ID in any supported format

        Returns:
            Standardized location ID
        """
        # Convert to string if needed
        loc_id = str(loc_id)

        # Handle depot conversions
        depot_mappings = {
            "1": "CD01",
            "CDA": "CD01",
            "2": "CD02",
            "CDB": "CD02",
            "3": "CD03",
            "CDC": "CD03",
        }

        if loc_id in depot_mappings:
            return depot_mappings[loc_id]

        # Handle numeric client IDs (if all digits and doesn't start with C or CD)
        if loc_id.isdigit():
            return f"C{int(loc_id):03d}"

        # Already standardized or unknown format - return as-is
        return loc_id

    def matrix_distance(self, loc1: str, loc2: str) -> float:
        """
        Look up distance from preloaded matrix.

        Args:
            loc1: Origin location ID (any supported format)
            loc2: Destination location ID (any supported format)

        Returns:
            Distance in kilometers

        Raises:
            KeyError: If location pair not found in matrix
            ValueError: If distance matrix not loaded
        """
        if self.distance_matrix is None:
            raise ValueError(
                "Distance matrix not loaded. Use --matrix argument with --method matrix."
            )

        # Normalize IDs to standardized format
        norm_loc1 = self.normalize_location_id(loc1)
        norm_loc2 = self.normalize_location_id(loc2)

        # Try direct lookup
        if norm_loc1 in self.distance_matrix and norm_loc2 in self.distance_matrix[norm_loc1]:
            return self.distance_matrix[norm_loc1][norm_loc2]

        # Try reverse for symmetric matrices
        if norm_loc2 in self.distance_matrix and norm_loc1 in self.distance_matrix[norm_loc2]:
            return self.distance_matrix[norm_loc2][norm_loc1]

        # Build helpful error message
        available_origins = list(self.distance_matrix.keys())[:10]
        error_msg = (
            f"Distance not found: {norm_loc1} → {norm_loc2}\n"
            f"  Original IDs: {loc1} → {loc2}\n"
        )

        if norm_loc1 not in self.distance_matrix:
            error_msg += f"  Origin '{norm_loc1}' not in matrix. Available origins: {available_origins}"
        else:
            available_dests = list(self.distance_matrix[norm_loc1].keys())[:10]
            error_msg += f"  Destination '{norm_loc2}' not available from '{norm_loc1}'. "
            error_msg += f"Available destinations: {available_dests}"

        raise KeyError(error_msg)

    def haversine_distance(self, loc1: str, loc2: str) -> float:
        """Calculate the distance between two locations using Haversine formula."""
        p1 = self.locations[loc1]
        p2 = self.locations[loc2]

        lat1, lon1 = p1["latitude"], p1["longitude"]
        lat2, lon2 = p2["latitude"], p2["longitude"]

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        earth_radius = 6371

        return earth_radius * c

    def geopy_distance(self, loc1: str, loc2: str) -> float:
        """Calculate distance using GeoPy."""
        p1 = self.locations[loc1]
        p2 = self.locations[loc2]

        coord1 = (p1["latitude"], p1["longitude"])
        coord2 = (p2["latitude"], p2["longitude"])

        return geodesic(coord1, coord2).kilometers

    def osrm_distance(self, loc1: str, loc2: str) -> float:
        """Calculate distance using OSRM API."""
        p1 = self.locations[loc1]
        p2 = self.locations[loc2]

        url = f"http://router.project-osrm.org/route/v1/driving/{p1['longitude']},{p1['latitude']};{p2['longitude']},{p2['latitude']}"

        try:
            response = requests.get(url)
            data = response.json()

            if "routes" in data and len(data["routes"]) > 0:
                # Distance in meters, convert to kilometers
                return data["routes"][0]["distance"] / 1000
            else:
                print(
                    f"OSRM API failed for {loc1} to {loc2}, falling back to Haversine"
                )
                return self.haversine_distance(loc1, loc2)
        except Exception as e:
            print(f"Error with OSRM API: {e}, falling back to Haversine")
            return self.haversine_distance(loc1, loc2)

    def calculate_distance(self, loc1: str, loc2: str) -> float:
        """Calculate distance using the selected method with caching."""
        # If using matrix method, skip caching (matrix is already a lookup table)
        if self.distance_method == "matrix":
            return self.matrix_distance(loc1, loc2)

        cache_key = f"{loc1}_{loc2}_{self.distance_method}"

        # Check cache first
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        # Calculate distance based on method
        if self.distance_method == "haversine":
            distance = self.haversine_distance(loc1, loc2)
        elif self.distance_method == "geopy":
            distance = self.geopy_distance(loc1, loc2)
        elif self.distance_method == "osrm":
            distance = self.osrm_distance(loc1, loc2)
        else:
            raise ValueError(f"Unknown distance method: {self.distance_method}")

        # Cache the result
        self.distance_cache[cache_key] = distance

        return distance

    def _get_column_value(self, row, *column_names):
        """Get value from row trying multiple column name variations."""
        for col_name in column_names:
            if col_name in row.index:
                return row[col_name]
        # If none found, raise error with all attempted names
        raise KeyError(f"None of these columns found: {column_names}")

    def validate_solution(self) -> Dict:
        """Validate the solution according to the specified requirements."""
        errors = []
        visited_clients = set()

        for idx, route in self.solution_df.iterrows():
            vehicle_id = route["VehicleId"]
            depot_id = route["DepotId"]

            # Handle column name variations for InitialLoad
            initial_load_str = self._get_column_value(route, "InitialLoad", "InitLoad")
            initial_load = int(initial_load_str)

            route_sequence = route["RouteSequence"].split("-")

            # Handle column name variations for ClientsServed
            clients_served_str = self._get_column_value(route, "ClientsServed", "Clients")
            clients_served = int(clients_served_str)

            # Handle column name variations for DemandsSatisfied
            demands_str = self._get_column_value(route, "DemandsSatisfied", "DemandSatisfied")
            demands_satisfied = [int(d) for d in demands_str.split("-")]

            # Fix: Use "V" prefix instead of "VEH" to match standardized format (V001, V002, etc.)
            vehicle_number = int(vehicle_id.replace("V", ""))
            vehicle_spec = self.vehicle_specs[vehicle_number]

            # Check 1: Route starts and ends at depot
            if route_sequence[0] != depot_id or route_sequence[-1] != depot_id:
                errors.append(
                    f"Route {vehicle_id} does not start and end at depot {depot_id}"
                )

            # Check 2: Vehicle capacity
            if initial_load > vehicle_spec["capacity"]:
                errors.append(
                    f"Route {vehicle_id} exceeds capacity: {initial_load} > {vehicle_spec['capacity']}"
                )

            # Check 3: Route range
            total_distance = 0
            for i in range(len(route_sequence) - 1):
                from_loc = route_sequence[i]
                to_loc = route_sequence[i + 1]

                # Check if locations exist before calculating distance
                if from_loc not in self.locations:
                    errors.append(
                        f"Route {vehicle_id} has invalid location: {from_loc}"
                    )
                    continue
                if to_loc not in self.locations:
                    errors.append(f"Route {vehicle_id} has invalid location: {to_loc}")
                    continue

                try:
                    distance = self.calculate_distance(from_loc, to_loc)
                    total_distance += distance
                except Exception as e:
                    errors.append(
                        f"Route {vehicle_id} distance calculation error: {str(e)}"
                    )

            if total_distance > vehicle_spec["range"]:
                errors.append(
                    f"Route {vehicle_id} exceeds range: {total_distance:.1f} > {vehicle_spec['range']}"
                )

            # Check 4: Track visited clients and demand satisfaction
            client_idx = 0
            for i, loc in enumerate(route_sequence):
                # Only count locations that start with C and are actual clients (not depots)
                if loc.startswith("C") and loc not in ["CDA", "CDB", "CDC"]:
                    if client_idx >= len(demands_satisfied):
                        errors.append(
                            f"Route {vehicle_id} has missing demand value for client {loc}"
                        )
                        continue

                    visited_clients.add(loc)
                    expected_demand = self.client_demands.get(loc, 0)
                    actual_demand = demands_satisfied[client_idx]

                    if actual_demand != expected_demand:
                        errors.append(
                            f"Route {vehicle_id} has incorrect demand for {loc}: {actual_demand} != {expected_demand}"
                        )

                    client_idx += 1

            # Check for duplicate client visits
            route_clients = [
                loc
                for loc in route_sequence
                if loc.startswith("C") and loc not in ["CDA", "CDB", "CDC"]
            ]
            if len(route_clients) != len(set(route_clients)):
                errors.append(f"Route {vehicle_id} has duplicate client visits")

            # Verify clients_served count
            if len(route_clients) != clients_served:
                errors.append(
                    f"Route {vehicle_id} clients_served mismatch: {len(route_clients)} != {clients_served}"
                )

        # Check 5: All clients visited
        all_clients = set(self.client_demands.keys())
        missing_clients = all_clients - visited_clients
        if missing_clients:
            for client in missing_clients:
                errors.append(f"Client {client} was not visited")

        # Save cache before returning (skip if using matrix method)
        if self.distance_method != "matrix":
            self.save_cache()

        return {"feasible": len(errors) == 0, "errors": errors}


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle Routing Solution Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Distance Calculation Methods:
  haversine : Great-circle distance (fast, approximate)
  geopy     : Geodesic distance (accurate, slower)
  osrm      : Road network distance via OSRM API (realistic, requires internet)
  matrix    : Precomputed distance matrix from file (fastest, requires --matrix)

Examples:
  python base_case_verification.py --method haversine
  python base_case_verification.py --method matrix --matrix distances.json
  python base_case_verification.py --solution verificacion_caso1.csv --verbose
        """,
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["haversine", "geopy", "osrm", "matrix"],
        default="haversine",
        help="Distance calculation method",
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default=None,
        help="Path to distance matrix file (.json or .csv). Required when --method matrix is used.",
    )
    parser.add_argument(
        "--solution",
        type=str,
        default="verificacion_caso1.csv",
        help="Path to solution file (default: verificacion_caso1.csv, fallback: solution.csv)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="distance_cache.json",
        help="Path to cache file (ignored when using matrix method)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # Validate matrix argument
    if args.method == "matrix" and not args.matrix:
        parser.error("--matrix argument is required when using --method matrix")

    if args.matrix and args.method != "matrix":
        print(
            "WARNING: --matrix provided but method is not 'matrix'. Matrix file will be ignored.\n"
        )

    try:
        print(
            f"Starting solution validation with {args.method} distance calculation..."
        )

        if args.method == "matrix":
            print(f"Using distance matrix from: {args.matrix}")
        else:
            print("Note: TotalDistance, TotalTime, and FuelCost columns are ignored")

        print(f"Reading solution from: {args.solution}\n")

        validator = SolutionValidator(
            distance_method=args.method,
            cache_file=args.cache,
            matrix_file=args.matrix,
            solution_file=args.solution,
        )
        result = validator.validate_solution()

        if result["feasible"]:
            print("\n✓ SOLUTION IS FEASIBLE!")
            print("All routes satisfy the requirements.")
        else:
            print("\n✗ SOLUTION IS INFEASIBLE!")
            print("Errors found:")
            for error in result["errors"]:
                print(f"  - {error}")

        if args.verbose:
            print("\nValidation completed successfully!")

    except Exception as e:
        print(f"\nError during validation: {str(e)}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
