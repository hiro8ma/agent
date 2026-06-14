"""Tool groups for the three supply-chain specialists.

Each specialist owns a disjoint group of tools. Grouping is the whole point of a
supervisor architecture: instead of one agent holding every tool (and a prompt that
must describe all of them), each specialist sees only the handful relevant to its
domain. Smaller tool sets mean shorter prompts, fewer mis-selections, and clearer
ownership — the cost is the supervisor's routing hop.

Every tool is a fake, offline implementation over a fictional supply network. No real
network call is made; any credential a real version would need is read from
``os.getenv`` and echoed only as a boolean presence flag, never embedded in source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Specialist:
    """A named worker and the names of the tools bound to it."""

    name: str
    description: str
    tool_names: tuple[str, ...]


# --- Inventory ---------------------------------------------------------------


class StockLevelArgs(BaseModel):
    sku: str = Field(description="Stock-keeping unit identifier, e.g. 'SKU-4471'.")


def check_stock_level(sku: str) -> str:
    """Return on-hand quantity and reorder point for a SKU across warehouses."""

    return (
        f"[inventory] {sku}: on_hand=0 reorder_point=120 status=STOCKOUT "
        f"warehouses=[WH-NORTH:0, WH-WEST:0]"
    )


class DemandForecastArgs(BaseModel):
    sku: str = Field(description="SKU to forecast demand for.")
    horizon_days: int = Field(default=30, description="Forecast horizon in days.")


def forecast_demand(sku: str, horizon_days: int = 30) -> str:
    """Project expected demand for a SKU over a horizon (fake time-series model)."""

    return f"[inventory] {sku}: forecast_units={horizon_days * 8} over {horizon_days}d, trend=rising"


class ReorderArgs(BaseModel):
    sku: str = Field(description="SKU to reorder.")
    quantity: int = Field(description="Units to request from the supplier.")


def place_reorder(sku: str, quantity: int) -> str:
    """Raise a replenishment purchase request for a SKU (warehouse system)."""

    return f"[inventory] reorder raised: {sku} x{quantity}, po=PO-{abs(hash(sku)) % 10000:04d}"


# --- Transportation ----------------------------------------------------------


class TrackShipmentArgs(BaseModel):
    shipment_id: str = Field(description="Shipment / tracking identifier, e.g. 'SHP-908'.")


def track_shipment(shipment_id: str) -> str:
    """Return current location and ETA for a shipment (carrier tracking)."""

    return f"[transport] {shipment_id}: location=HUB-CENTRAL eta=+2d status=DELAYED reason=weather"


class ArrangeTransportArgs(BaseModel):
    origin: str = Field(description="Origin hub or warehouse code.")
    destination: str = Field(description="Destination hub or warehouse code.")
    priority: str = Field(default="standard", description="'standard' or 'expedited'.")


def arrange_transport(origin: str, destination: str, priority: str = "standard") -> str:
    """Book a lane between two nodes at a service level (carrier booking)."""

    has_key = bool(os.getenv("CARRIER_API_KEY"))
    return (
        f"[transport] booked {origin}->{destination} priority={priority} "
        f"booking=BK-7782 carrier_key_present={has_key}"
    )


class RerouteArgs(BaseModel):
    shipment_id: str = Field(description="Shipment to reroute around a disruption.")


def reroute_shipment(shipment_id: str) -> str:
    """Compute an alternate route around a disruption (routing engine)."""

    return f"[transport] {shipment_id}: rerouted via HUB-EAST, eta_recovered=+1d"


# --- Supplier ----------------------------------------------------------------


class SupplierScoreArgs(BaseModel):
    supplier_id: str = Field(description="Supplier identifier, e.g. 'SUP-12'.")


def score_supplier(supplier_id: str) -> str:
    """Return a performance scorecard for a supplier (vendor analytics)."""

    return (
        f"[supplier] {supplier_id}: on_time=82% defect_rate=1.4% lead_time=11d "
        f"rating=B capacity=available"
    )


class ComplianceArgs(BaseModel):
    supplier_id: str = Field(description="Supplier to check compliance status for.")


def check_compliance(supplier_id: str) -> str:
    """Check certification and compliance status for a supplier (audit registry)."""

    return f"[supplier] {supplier_id}: iso=valid sanctions_clear=true audit=2025-11 status=COMPLIANT"


class AlternateSupplierArgs(BaseModel):
    sku: str = Field(description="SKU to find an alternate qualified supplier for.")


def find_alternate_supplier(sku: str) -> str:
    """List qualified backup suppliers for a SKU (sourcing catalog)."""

    return f"[supplier] {sku}: alternates=[SUP-31 rating=A lead=7d, SUP-44 rating=B lead=9d]"


def _tool(func: object, name: str, schema: type[BaseModel], description: str) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,  # type: ignore[arg-type]
        name=name,
        description=description,
        args_schema=schema,
    )


CHECK_STOCK_LEVEL = _tool(
    check_stock_level,
    "check_stock_level",
    StockLevelArgs,
    "Look up on-hand quantity, reorder point, and stockout status for a SKU.",
)
FORECAST_DEMAND = _tool(
    forecast_demand,
    "forecast_demand",
    DemandForecastArgs,
    "Project expected demand for a SKU over a horizon in days.",
)
PLACE_REORDER = _tool(
    place_reorder,
    "place_reorder",
    ReorderArgs,
    "Raise a replenishment purchase request for a SKU and quantity.",
)
TRACK_SHIPMENT = _tool(
    track_shipment,
    "track_shipment",
    TrackShipmentArgs,
    "Get current location, ETA, and delay status for a shipment.",
)
ARRANGE_TRANSPORT = _tool(
    arrange_transport,
    "arrange_transport",
    ArrangeTransportArgs,
    "Book transport on a lane between two nodes at a service level.",
)
REROUTE_SHIPMENT = _tool(
    reroute_shipment,
    "reroute_shipment",
    RerouteArgs,
    "Compute an alternate route for a shipment around a disruption.",
)
SCORE_SUPPLIER = _tool(
    score_supplier,
    "score_supplier",
    SupplierScoreArgs,
    "Return a performance scorecard (on-time, defects, lead time) for a supplier.",
)
CHECK_COMPLIANCE = _tool(
    check_compliance,
    "check_compliance",
    ComplianceArgs,
    "Check certification, sanctions, and audit/compliance status for a supplier.",
)
FIND_ALTERNATE_SUPPLIER = _tool(
    find_alternate_supplier,
    "find_alternate_supplier",
    AlternateSupplierArgs,
    "List qualified backup suppliers for a SKU when the primary is at risk.",
)


SPECIALISTS: list[Specialist] = [
    Specialist(
        name="inventory",
        description=(
            "Stock levels, warehouse on-hand, demand forecasting, and replenishment / "
            "reorder decisions for SKUs."
        ),
        tool_names=("check_stock_level", "forecast_demand", "place_reorder"),
    ),
    Specialist(
        name="transportation",
        description=(
            "Shipment tracking, transport booking, ETAs, delays, and rerouting around "
            "logistics disruptions."
        ),
        tool_names=("track_shipment", "arrange_transport", "reroute_shipment"),
    ),
    Specialist(
        name="supplier",
        description=(
            "Supplier performance scoring, certification / compliance checks, and "
            "sourcing alternate vendors."
        ),
        tool_names=("score_supplier", "check_compliance", "find_alternate_supplier"),
    ),
]

SPECIALISTS_BY_NAME: dict[str, Specialist] = {s.name: s for s in SPECIALISTS}

ALL_TOOLS: list[StructuredTool] = [
    CHECK_STOCK_LEVEL,
    FORECAST_DEMAND,
    PLACE_REORDER,
    TRACK_SHIPMENT,
    ARRANGE_TRANSPORT,
    REROUTE_SHIPMENT,
    SCORE_SUPPLIER,
    CHECK_COMPLIANCE,
    FIND_ALTERNATE_SUPPLIER,
]

TOOLS_BY_NAME: dict[str, StructuredTool] = {t.name: t for t in ALL_TOOLS}


def tools_for(specialist: str) -> list[StructuredTool]:
    """Return the tool group bound to a specialist (its disjoint slice of ALL_TOOLS)."""

    spec = SPECIALISTS_BY_NAME.get(specialist)
    if spec is None:
        return []
    return [TOOLS_BY_NAME[name] for name in spec.tool_names]


def run_tool(name: str, args: dict[str, object]) -> str:
    """Execute a tool by name with keyword args (fake, offline)."""

    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return f"(no such tool: {name})"
    return str(tool.invoke(args))
