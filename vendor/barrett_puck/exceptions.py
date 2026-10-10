class PuckError(Exception):
    """Base class for all barrett_puck errors."""


class PuckTimeoutError(PuckError):
    """The puck did not reply within the allotted time."""


class PuckProtocolError(PuckError):
    """A received frame could not be decoded."""


class PuckStateError(PuckError):
    """The puck is in the wrong state for the requested operation."""


class SdoAbortError(PuckError):
    """The puck aborted an SDO transfer."""

    #: Common CiA-301 abort codes.
    CODES = {
        0x05030000: "toggle bit not alternated",
        0x05040000: "SDO protocol timed out",
        0x05040001: "client/server command specifier not valid",
        0x06010000: "unsupported access to an object",
        0x06010001: "attempt to read a write-only object",
        0x06010002: "attempt to write a read-only object",
        0x06020000: "object does not exist in the object dictionary",
        0x06040041: "object cannot be mapped to the PDO",
        0x06040042: "mapped objects exceed PDO length",
        0x06070010: "data type / length of service parameter does not match",
        0x06090011: "sub-index does not exist",
        0x06090030: "value range of parameter exceeded",
        0x08000000: "general error",
        0x08000020: "data cannot be transferred or stored",
        0x08000022: "data cannot be transferred (present device state)",
    }

    def __init__(self, node: int, index: int, sub: int, code: int):
        self.node = node
        self.index = index
        self.sub = sub
        self.code = code
        meaning = self.CODES.get(code, "unknown abort code")
        super().__init__(
            f"node {node}: SDO abort 0x{code:08X} ({meaning}) "
            f"accessing 0x{index:04X},{sub}"
        )
