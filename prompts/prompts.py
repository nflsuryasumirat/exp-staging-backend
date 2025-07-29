# PROMPT_REMOVE = "empty room, clean walls, professional real estate photography, high quality, sharp focus, hyper-realistic"
PROMPT_REMOVE = "empty room, empty interior, natural lighting, clean walls, professional real estate photography, high quality, sharp focus, hyper-realistic"
PROMPT_NEG_REMOVE = "furnitures, chairs, tables, sofa, bed, clutter, objects, people, text, watermark, low-quality, tv, lamps, paintings, blurry, animals, furnishings, decorations, piano, musical instruments, stairs"

PROMPT_ADD = lambda room, layout, style: f"A sophisticated {layout} {room} with furnishings in {style} style that match the room's architecture and lighting. Render in photorealistic style with proper scale, shadows, and reflections. Ensure all items are proportionate, logically placed, and blend seamlessly into the space."
PROMPT_NEG_ADD = "clutter, people, text, watermark, low quality, outdoor, blurry, animals"
