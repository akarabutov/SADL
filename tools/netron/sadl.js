// SPDX-License-Identifier: MIT
// Copyright (c) 2026 KHU-VCP-Lab

const sadl = {};

sadl.DataType = {
    INT32: 0,  // int32_t
    FLOAT: 1,  // float32
    INT16: 2,  // int16_t
};

sadl.DataTypeName = new Map([
    [sadl.DataType.INT32, 'int32'],
    [sadl.DataType.FLOAT, 'float32'],
    [sadl.DataType.INT16, 'int16'],
]);

sadl.OpSpec = {
  Const:            { id:  1, payload: true  },
  Placeholder:      { id:  2, payload: true  },
  Identity:         { id:  3, payload: false },
  BiasAdd:          { id:  4, payload: false },
  MaxPool:          { id:  5, payload: true  },
  MatMul:           { id:  6, payload: true  },
  Reshape:          { id:  7, payload: false },
  Relu:             { id:  8, payload: false },
  Conv2D:           { id:  9, payload: true  },
  Add:              { id: 10, payload: false },
  ConcatV2:         { id: 11, payload: false },
  Mul:              { id: 12, payload: true  },
  Maximum:          { id: 13, payload: false },
  LeakyReLU:        { id: 14, payload: false },
  Transpose:        { id: 15, payload: false },
  Flatten:          { id: 16, payload: true  },
  Shape:            { id: 17, payload: false },
  Expand:           { id: 18, payload: false },
  Conv2DTranspose:  { id: 19, payload: true  },
  Slice:            { id: 20, payload: true  },
  PReLU:            { id: 21, payload: false },
  ScatterND:        { id: 22, payload: false },
  GridSample:       { id: 23, payload: true  },
  Resize:           { id: 24, payload: true  },
  Compare:          { id: 25, payload: true  },
  Where:            { id: 26, payload: false },
  Minimum:          { id: 27, payload: false },
  AveragePool:      { id: 28, payload: true  },
  ReduceMean:       { id: 29, payload: true  },
  Tile:             { id: 30, payload: false },
  Pad:              { id: 31, payload: true  },
  Floor:            { id: 32, payload: false },
  Sub:              { id: 33, payload: false },
  Abs:              { id: 34, payload: false },
  Depth2Space:      { id: 35, payload: true  },
  Gather:           { id: 36, payload: true  },
  GatherElements:   { id: 37, payload: true  },
  GlobalAveragePool:{ id: 38, payload: false },
  BitShift:         { id: 39, payload: true  },
  Clip:             { id: 40, payload: false },
};

sadl.OpCode = Object.fromEntries(
  Object.entries(sadl.OpSpec).map(([name, spec]) => [name, spec.id])
);

// Reverse lookup for op_id -> name
sadl.OpName = Object.fromEntries(
  Object.entries(sadl.OpCode).map(([name, id]) => [id, name])
);

sadl.ModelFactory = class {
    async match(context) {
        const id = (context.identifier || '').toLowerCase();
        if (!id.endsWith('.sadl')) {
            return null;
        }
        const reader = await sadl.BinaryReader.open(context);
        return reader ? context.set(reader.name, reader) : null;
    }

    async open(context) {
        const target = context.value;   // BinaryReader
        await target.read();
        const metadata = await sadl.Metadata.open(context); // internal resource
        return new sadl.Model(metadata, target);
    }
};

sadl.Model = class {
    constructor(metadata, target) {
        const model = target.model;
        this._modules = [];
        this._format = target.format;
        this._producer = null;
        this._source = null;
        this._domain = null;
        this._version = (model && typeof model.version !== 'undefined') ? String(model.version) : '';
        this._description = null;
        this._metadata = [];
        this._imports = null;
        // Build graph via Context
        const imageFormat = '';
        const imports = new Map();
        const context = new sadl.Context.Model(metadata, target.locations, imageFormat, imports, model);
        const graph = context.graph(null);
        if (graph) {
            this._modules.push(graph);
        }
        this._functions = context.functions;
    }

    get format() { return this._format; }
    get version() { return this._version; }
    get imports() { return this._imports; }
    get producer() { return this._producer; }
    get source() { return this._source; }
    get domain() { return this._domain || null; }
    get description() { return this._description || null; }
    get metadata() { return this._metadata; }
    get modules() { return this._modules; }
    get functions() { return this._functions; }
};

sadl.Graph = class {
    constructor(context, graph) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._name = graph?.name || null;
        this._description = graph?.doc_string || graph?.description || '';
        this._metadata = (graph?.metadata_props || []).map((m) => new sadl.Argument(m.key, m.value));

        const gctx = new sadl.Context.Graph(context, graph);

        // Build nodes
        gctx.push(graph?.node || []);
        this._nodes = gctx.pop();

        // Visible graph inputs
        for (const input of (graph?.input || [])) {
            const value = gctx.value(input.name);
            value.metadata = input.metadata || [];
            if (!value.initializer) {
                this._inputs.push(new sadl.Argument(input.name, [value]));
            }
        }

        // Visible graph outputs
        for (const output of (graph?.output || [])) {
            const value = gctx.value(output.name);
            value.metadata = output.metadata || [];
            if (!value.initializer) {
                this._outputs.push(new sadl.Argument(output.name, [value]));
            }
        }
    }

    get name() { return this._name; }
    get description() { return this._description; }
    get metadata() { return this._metadata; }
    get inputs() { return this._inputs; }
    get outputs() { return this._outputs; }
    get nodes() { return this._nodes; }
    toString() { return `graph(${this.name})`; }
};


sadl.Argument = class {
    constructor(name, value, type, description, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.description = description || null;
        this.visible = visible !== false;
    }
};

sadl.Value = class {
    constructor(name, type, initializer, annotation, description) {
        if (typeof name !== 'string') {
            throw new Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
        this._description = description || '';
        this._quantization = null;
    }

    get name() { return this._name; }
    get type() { return this._type; }
    get description() { return this._description; }
    get quantization() { return this._quantization; }
    get initializer() { return this._initializer; }
};

sadl.Node = class {
    constructor(context, layer) {
        if (!layer || typeof layer !== 'object') {
            throw new Error('Invalid SADL layer object.');
        }

        // type
        const typeId = layer.type;
        if (typeof typeId !== 'string' || typeId.length === 0) {
            throw new Error(`Invalid SADL layer type: ${JSON.stringify(typeId)}.`);
        }
        this.type = { name: typeId, module: 'sadl', identifier: typeId };

        // basic info
        this.name = (typeof layer.name === 'string') ? layer.name : '';
        this.description = (typeof layer.description === 'string') ? layer.description : '';
        this.metadata = Array.isArray(layer.metadata) ? layer.metadata : [];

        const layerInfo = `${this.type.name}${this.name ? ` '${this.name}'` : ''}`;

        // inputs
        const inList = layer.inputs ?? [];
        if (!Array.isArray(inList)) {
            throw new Error(`Invalid SADL layer.inputs in ${layerInfo}: expected array.`);
        }
        this.inputs = inList.map((v, i) => {
            if (typeof v !== 'string') {
                throw new Error(`Invalid SADL input[${i}] in ${layerInfo}: ${JSON.stringify(v)} (expected string).`);
            }
            const argName = (Array.isArray(layer.inputNames) && typeof layer.inputNames[i] === 'string')
                ? layer.inputNames[i]
                : i.toString();
            return new sadl.Argument(argName, [context.value(v)]);
        });

        // outputs
        const outList = layer.outputs ?? [];
        if (!Array.isArray(outList) || outList.length === 0) {
            throw new Error(`Missing/invalid SADL layer.outputs for ${layerInfo}.`);
        }
        this.outputs = outList.map((v, i) => {
            if (typeof v !== 'string') {
                throw new Error(`Invalid SADL output[${i}] in ${layerInfo}: ${JSON.stringify(v)} (expected string).`);
            }
            const argName = (Array.isArray(layer.outputNames) && typeof layer.outputNames[i] === 'string')
                ? layer.outputNames[i]
                : i.toString();
            return new sadl.Argument(argName, [context.value(v)]);
        });

        // attributes
        const attrs = layer.attributes ?? [];
        if (!Array.isArray(attrs)) {
            throw new Error(`Invalid SADL layer.attributes in ${layerInfo}: expected array.`);
        }
        this.attributes = attrs.map((attr) => {
            if (!attr || typeof attr !== 'object' || typeof attr.name !== 'string') {
                throw new Error(`Invalid attribute in ${layerInfo}: ${JSON.stringify(attr)}.`);
            }
            return new sadl.Argument(attr.name, attr.value, attr.type);
        });
    }
};

sadl.Tensor = class {
    constructor(context, tensor, category) {
        this._category = category || null;
        this._name = (tensor && typeof tensor.name === 'string') ? tensor.name : '';
        const dims = (tensor && Array.isArray(tensor.dims)) ? tensor.dims.slice() : [];
        let dtype = 'float32';
        try { dtype = context.createDataType(tensor && tensor.dtype); } catch { /* ignore */ }
        this._type = context.createTensorType(dtype, dims);
        this._location = null;
        this._encoding = '|';
        this._description = null;
        // preserve existing description if present
        if (tensor && typeof tensor.description === 'string' && tensor.description.length > 0) {
            this._description = tensor.description;
        }

        // append quantizer info if present
        if (tensor && tensor.quantizer !== null && tensor.quantizer !== undefined) {
            const q = `quantizer=${tensor.quantizer}`;
            this._description = this._description ? `${this._description}, ${q}` : q;
        }

        // Load data if present
        if (tensor && (Array.isArray(tensor.data) || ArrayBuffer.isView(tensor.data))) {
            this._data = tensor.data;
            if (this._data.length === 0) { this._data = undefined; }
            return;
        }

        // Load raw_data if present
        this._data = undefined;
        if (tensor && (tensor.raw_data instanceof Uint8Array || tensor.raw_data instanceof ArrayBuffer)) {
            const buffer = (tensor.raw_data instanceof Uint8Array)
                ? tensor.raw_data.buffer.slice(
                    tensor.raw_data.byteOffset,
                    tensor.raw_data.byteOffset + tensor.raw_data.byteLength
                )
                : tensor.raw_data;

            switch (dtype) {
                case 'float32': this._data = new Float32Array(buffer); break;
                case 'int32':   this._data = new Int32Array(buffer);   break;
                case 'int16':   this._data = new Int16Array(buffer);   break;
                default:
                    this._data = new Uint8Array(buffer);
                    this._encoding = '<';
                    break;
            }
            if (this._data.length === 0) { this._data = undefined; }
        }
    }

    peek() { return true; }
    async read() { }

    get name() { return this._name; }
    get category() { return this._category; }
    get description() { return this._description; } 
    get encoding() { return this._encoding; }
    get location() { return this._location; }
    get type() { return this._type; }
    get values() { return this._data; }
};

sadl.TensorType = class {
    constructor(dataType, shape, layout, denotation) {
        this._dataType = dataType;
        this._shape = shape;
        this._layout = layout || null;
        this._denotation = denotation || null;
    }

    get dataType() { return this._dataType; }
    get shape() { return this._shape; }
    get layout() { return this._layout; }
    get denotation() { return this._denotation; }
    toString() {
        return `${this.dataType}${this.shape ? this.shape.toString() : ''}`;
    }
};

sadl.TensorShape = class {
    constructor(dimensions) {
        const dims = Array.isArray(dimensions) ? dimensions : [];
        this._dimensions = dims.map((dim) => (typeof dim === 'bigint' ? Number(dim) : dim));
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length === 0) return '';
        const parts = this._dimensions.map((dim) =>
          Number.isInteger(dim) ? String(dim) : '?'
        );
        return `[${parts.join(',')}]`;
    }
};

sadl.Context = {};

sadl.Context.Model = class {
    constructor(metadata, locations, imageFormat, imports, model) {
        this._metadata = metadata;
        this._locations = locations || new Map();
        this._imports = imports || new Map();
        this._model = model || null;
        this._functions = [];
    }

    graph(value) {
        const g = value || (this._model && this._model.graph) || null;
        return g ? new sadl.Graph(this, g) : null;
    }

    get functions() {
        return this._functions;
    }

    location(name) {
        return (this._locations && this._locations.has(name)) ? this._locations.get(name) : null;
    }

    initializer(/* name */) {
        return null;
    }

    type(domain, name, overload) {
        const d = domain || 'sadl';
        if (this._metadata && typeof this._metadata.type === 'function') {
            const t = this._metadata.type(d, name);
            if (t) { return t; }
        }
        return {
            name,
            module: d,
            overload: overload || '',
            identifier: name
        };
    }

    attribute(domain, type, overload, name) {
        const meta = this.type(domain || 'sadl', type, overload);
        if (meta && Array.isArray(meta.attributes)) {
            return meta.attributes.find((a) => a.name === name) || null;
        }
        return null;
    }

    decodeText(value) {
        if (typeof value === 'string') {
            return value;
        }
        this._decoder = this._decoder || new TextDecoder('utf-8');
        return this._decoder.decode(value);
    }

    createAttribute(attr, metadata) {
        const name = attr.name;
        const value = attr.value;
        const type = attr.type || null;
        let visible = true;
        if (metadata && metadata.default !== undefined && value === metadata.default) {
            visible = false;
        }
        return new sadl.Argument(name, value, type, attr.description, visible);
    }

    createDataType(value) {
        if (Number.isInteger(value)) {
            const dt = sadl.DataTypeName.get(value);
            if (dt) return dt;
            throw (sadl.Error ? new sadl.Error(`Unsupported data type '${value}'.`)
                             : new Error(`Unsupported data type '${value}'.`));
        }

        if (typeof value === 'string') {
            const k = value.toLowerCase();
            if (k === 'float' || k === 'float32') return 'float32';
            if (k === 'int' || k === 'int32' || k === 'i32') return 'int32';
            if (k === 'int16' || k === 'i16') return 'int16';
        }

        throw (sadl.Error ? new sadl.Error(`Unsupported data type '${JSON.stringify(value)}'.`)
                         : new Error(`Unsupported data type '${JSON.stringify(value)}'.`));
    }

    createLocation(/* value */) {
        return '';
    }

    createTensorType(dataType, shape, layout, denotation) {
        const dt = this.createDataType(dataType);
        return new sadl.TensorType(dt, new sadl.TensorShape(shape || []), layout, denotation);
    }
};

// not implemented yet
sadl.Metadata = class {
    static async open(context) {
        if (!sadl.Metadata._metadata) {
            let data = null;
            try {
                data = await context.request('sadl-metadata.json');
            } catch {
                // ignore
            }
            sadl.Metadata._metadata = new sadl.Metadata(data);
        }
        return sadl.Metadata._metadata;
    }

    constructor(data) {
        this._types = new Map();
        if (!data) {
            return;
        }

        let entries = null;
        try {
            entries = JSON.parse(data);
        } catch {
            return;
        }

        if (!Array.isArray(entries)) {
            return;
        }

        for (const type of entries) {
            if (!type || typeof type !== 'object') {
                continue;
            }
            if (!this._types.has(type.module)) {
                this._types.set(type.module, new Map());
            }
            const byName = this._types.get(type.module);
            if (!byName.has(type.name)) {
                byName.set(type.name, []);
            }
            byName.get(type.name).push(type);
        }
    }

    type(domain, name) {
        const module = domain || 'sadl';
        let current = null;

        const byName = this._types.get(module);
        const list = byName ? byName.get(name) : null;
        if (!list) {
            return null;
        }

        for (const t of list) {
            const v = (t && t.version != null) ? t.version : 0;
            const best = current ? (current.version ?? -1) : -1;
            if (v > best) {
                current = t;
            }
        }
        return current;
    }
};

sadl.Context.Graph = class {
    constructor(context, graph) {
        this._context = context;
        this._tensors = new Map();
        this._values = new Map();
        this._nodes = [];

        if (Array.isArray(graph?.node)) {
            for (const node of graph.node) {
                node.type       = node.type || 'Op';
                node.inputs     = Array.isArray(node.inputs) ? node.inputs : [];
                node.outputs    = Array.isArray(node.outputs) ? node.outputs : [];
                node.attributes = node.attributes || [];
                node.param      = node.param || {};
            }
        }
    }

    tensor(name) {
        if (typeof name !== 'string') {
            throw new Error(`Invalid SADL tensor identifier '${JSON.stringify(name)}'.`);
        }
        if (!this._tensors.has(name)) {
            this._tensors.set(name, { name, initializer: null });
        }
        return this._tensors.get(name);
    }

    value(name) {
        if (!this._values.has(name)) {
            const t = this.tensor(name);
            const type = t.initializer ? t.initializer.type : (t.type || null);
            this._values.set(name, new sadl.Value(name, type, t.initializer, t.annotation, t.description));
        }
        return this._values.get(name);
    }

    push(nodes) {
        // Pass 1: materialize dense Const into initializers so later nodes see them
        for (const node of nodes) {
            if (node.type !== 'Const' || !node.const_data) continue;

            const { raw_data, dims, dtype, quantizer, isSparse, sparse } = node.const_data;
            const tensorName = node.outputs[0];
            const sparseFlag =
                ((isSparse !== null && isSparse !== undefined && isSparse !== 0) ||
                 (sparse !== null && sparse !== undefined));

            // Sparse const -> NOT create initializer
            if (sparseFlag) {
                continue;
            }

            // Dense const -> initializer
            const initTensor = new sadl.Tensor(this._context, {
                name: tensorName,
                dims: dims,
                dtype: dtype,
                raw_data: raw_data,
                quantizer: quantizer
            }, 'Constant');

            const t = this.tensor(tensorName);
            t.initializer = initTensor;
            t.type = initTensor.type;

            // If a Value was created earlier for this tensor, drop it so next lookup reflects initializer/annotation
            this._values.delete(tensorName);
        }

        // Pass 2: build nodes and skip dense Const nodes turned into initializers
        for (const node of nodes) {
            if (node.type === 'Const' && node.const_data) {
                const { isSparse, sparse } = node.const_data;
                const sparseFlag =
                    ((isSparse !== null && isSparse !== undefined && isSparse !== 0) ||
                     (sparse !== null && sparse !== undefined));

                // dense Const already materialized as initializer -> hide node
                if (!sparseFlag) {
                    continue;
                }
            }

            this._nodes.push(new sadl.Node(this, node));
        }
    }


    pop() {
        return this._nodes;
    }
};

sadl.ops = new Map();

sadl.registerOp = (key, fn) => {
    let id = key;
    if (typeof key === 'string') {
        if (!sadl.OpCode || sadl.OpCode[key] == null) {
            throw new sadl.Error(`Cannot register decoder: unknown OpCode '${key}'.`);
        }
        id = sadl.OpCode[key];
    }
    if (typeof id !== 'number' || !Number.isInteger(id)) {
        throw new sadl.Error(`Cannot register decoder: invalid key '${key}'.`);
    }
    if (typeof fn !== 'function') {
        throw new sadl.Error(`Decoder for op ${key} must be a function.`);
    }
    sadl.ops.set(id, fn);
};

sadl.getOp = (op_id, layer_id, offset) => {
  if (!Number.isInteger(op_id)) {
    throw new sadl.Error(`Cannot get op: invalid op_id '${String(op_id)}'.`);
  }

  const decoder = sadl.ops.get(op_id);
  if (decoder) return decoder;

  const opname = (sadl.OpName && sadl.OpName[op_id]) ? sadl.OpName[op_id] : `Op${op_id}`;
  const where =
    (layer_id !== undefined || offset !== undefined)
      ? ` at layer_id=${layer_id ?? '?'}${offset !== undefined ? `, offset=${offset}` : ''}`
      : '';

  throw new sadl.Error(
    `No decoder for op_id=${op_id} (${opname})${where}. ` +
    `Register one with sadl.registerOp(${op_id}, fn).`
  );
};

sadl.BinaryReader = class {

  static async open(context) {
    const id = context.identifier || '';
    const ext = id.includes('.') ? id.split('.').pop().toLowerCase() : '';
    const stream = context.stream;
    if (!stream || stream.length < 8) return null;

    const head = stream.peek(8);
    const magic = String.fromCharCode(head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7]);

    if (magic.startsWith('SADL')) return new sadl.BinaryReader(context);
    if (ext === 'sadl') return new sadl.BinaryReader(context);
    return null;
  }

  constructor(context) {
    this.name = 'sadl.binary';
    this.context = context;
    this.locations = new Map();
    this.model = null;
    this.format = null;
    this._version = null;
  }

  async read() {
    const stream = this.context.stream;
    const bytes = stream.peek();
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const textDec = new TextDecoder('utf-8');

    // One cursor used for: header + layer prefix + layer payload
    const cursor = {
      pos: 0,
      view,
      bytes,
      _dec: textDec,

      ensure(n) {
        if (this.pos + n > this.view.byteLength) {
          throw new sadl.Error(`SADL: unexpected EOF at ${this.pos}, need ${n} more bytes.`);
        }
      },

      readI32() { this.ensure(4); const v = this.view.getInt32(this.pos, true); this.pos += 4; return v; },
      readI16() { this.ensure(2); const v = this.view.getInt16(this.pos, true); this.pos += 2; return v; },
      readU8()  { this.ensure(1); const v = this.view.getUint8(this.pos);      this.pos += 1; return v; },

      readBytes(n) {
        this.ensure(n);
        const u = new Uint8Array(this.bytes.buffer, this.bytes.byteOffset + this.pos, n);
        this.pos += n;
        return u;
      },

      readStringN(n) {
        const u = this.readBytes(n);
        let end = u.length;
        while (end > 0 && u[end - 1] === 0) end--;
        return this._dec.decode(u.subarray(0, end));
      },

      readAscii(n) {
        const u = this.readBytes(n);
        return String.fromCharCode(...u);
      }
    };

    // Header
    const magic = cursor.readAscii(8);
    if (!magic.startsWith('SADL')) throw new sadl.Error('SADL: bad magic.');
    this._version = magic.slice(4);
    const version_num = Number.parseInt(this._version, 10) || 0;

    // Model type (TensorInternalType): 0=int32, 1=float32, 2=int16
    const type_model = cursor.readI32();

    // SADL0005+: optional model info string (int16 length + bytes)
    let model_info = '';
    if (version_num >= 5) {
      const info_len = cursor.readI16();
      if (info_len > 0) model_info = cursor.readStringN(info_len);
    }

    const nb_layers = cursor.readI32();

    const num_model_inputs = cursor.readI32();
    const inputs_id = new Array(num_model_inputs);
    for (let i = 0; i < num_model_inputs; i++) inputs_id[i] = cursor.readI32();

    const num_model_outputs = cursor.readI32();
    const outputs_id = new Array(num_model_outputs);
    for (let i = 0; i < num_model_outputs; i++) outputs_id[i] = cursor.readI32();

    // SADL0005+: 1-byte sparsityComputed flag (viewer can ignore)
    if (version_num >= 5) {
      cursor.readU8();
    }

    // Naming for tensors referenced by numeric ids in the stream.
    const tensorName = (id) => 't' + id;

    const graph = {
      name: '',
      doc_string: model_info || '',
      input:  inputs_id.map((id) => ({ name: tensorName(id) })),
      output: outputs_id.map((id) => ({ name: tensorName(id) })),
      node: []
    };

    // Layer loop
    for (let li = 0; li < nb_layers; li++) {
      const layerOffset = cursor.pos;

      const layer_id  = cursor.readI32();
      const op_id     = cursor.readI32();
      const name_size = cursor.readI32();

      const name = cursor.readStringN(name_size);

      const num_layer_inputs = cursor.readI32();
      const input_ids = new Array(num_layer_inputs);
      for (let i = 0; i < num_layer_inputs; i++) {
        input_ids[i] = cursor.readI32();
      }

      const decoder = sadl.getOp(op_id, layer_id, layerOffset);
      if (!decoder) {
        const opname = (sadl.OpName && sadl.OpName[op_id]) ? sadl.OpName[op_id] : `Op${op_id}`;
        throw new sadl.Error(
          `No decoder for op_id=${op_id} (${opname}) at layer_id=${layer_id}, offset=${layerOffset}. ` +
          `Register one with sadl.registerOp(${op_id}, fn).`
        );
      }

      const payloadStart = cursor.pos;

      const result = decoder({
        layer_id,
        op_id,
        name,
        cursor,
        tensorName,
        type_model,
        version: this._version,
        inputs: input_ids
      });

      if (!result || typeof result.pos !== 'number' || result.pos < cursor.pos) {
        throw new sadl.Error(
          `Decoder for op_id=${op_id} returned invalid pos (layer_id=${layer_id}, payloadStart=${payloadStart}, cursor.pos=${cursor.pos}, result.pos=${result && result.pos}).`
        );
      }

      // Allow decoder to "skip" forward by returning a larger pos than it actually read.
      cursor.pos = result.pos;

      const node = result.node || {};
      const op_type = (node.type && typeof node.type === 'string')
        ? node.type
        : (sadl.OpName && sadl.OpName[op_id]) ? sadl.OpName[op_id] : ('Op' + op_id);

      const n = {
        name: name || node.name || ('layer_' + layer_id),
        type: op_type,
        inputs:  (node.input  || []).map((t) => typeof t === 'number' ? tensorName(t) : t),
        outputs: (node.output || []).map((t) => typeof t === 'number' ? tensorName(t) : t),
        attributes: [{ name: 'layer_id', value: layer_id, type: 'int32' }].concat(node.attributes || []),
        const_data: node.const_data
      };

      graph.node.push(n);
    }

    this.model = { version: this._version, type_model, graph };
    this.format = `SADL v${this._version}`;
  }
};

sadl.decode = {
    readI32Array(cursor, n) {
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = cursor.readI32();
        return out;
    },
    readBytes(cursor, n) {
        return cursor.readBytes(n);
    },
    readStringN(cursor, n) {
        return cursor.readStringN(n);
    },
    makeNode({ name, type, input = [], output = [], attributes = [] }) {
        return { name, type, input, output, attributes };
    },
    needSpec({ layer_id, op_id, name }, msg = 'decoder needs payload spec') {
        const opname = (sadl.OpName && sadl.OpName[op_id]) ? sadl.OpName[op_id] : `Op${op_id}`;
        throw new sadl.Error(`[${opname}] layer_id=${layer_id} (“${name}”): ${msg}.`);
    }
};

sadl.decode.readCountedI32 = (cursor) => {
  const n = cursor.readI32();
  const v = new Array(n);
  for (let i = 0; i < n; i++) v[i] = cursor.readI32();
  return v;
};
sadl.decode.product = (dims) => (dims && dims.length
  ? dims.reduce((a, b) => a * (typeof b === 'bigint' ? Number(b) : b), 1)
  : 1);
sadl.decode.dtypeSize = (dtype) => (dtype === 1 ? 4 : dtype === 0 ? 4 : dtype === 2 ? 2 : dtype === 3 ? 1 : 4);
sadl.decode.inputsToNames = (ctx) => (ctx.inputs || []).map((id) => ctx.tensorName(id));
sadl.decode.outName = (ctx) => ctx.tensorName(ctx.layer_id);

sadl.decode.nodeResult = (ctx, type, attributes = [], opts = {}) => ({
  node: sadl.decode.makeNode({
    name: ctx.name,
    type,
    input: (opts.input !== undefined) ? opts.input : sadl.decode.inputsToNames(ctx),
    output: (opts.output !== undefined) ? opts.output : [sadl.decode.outName(ctx)],
    attributes,
  }),
  pos: ctx.cursor.pos,
});

sadl.decode.decodeOp = (type, parse, opts = {}) => (ctx) => {
  const parsed = parse(ctx);

  const attributes = Array.isArray(parsed)
    ? parsed
    : (parsed && Array.isArray(parsed.attributes) ? parsed.attributes : []);

  const result = sadl.decode.nodeResult(ctx, type, attributes, opts);

  if (parsed && typeof parsed.post === 'function') {
    parsed.post(result.node, ctx);
  }

  return result;
};

// decoders
const dec = Object.create(null);

// Const
dec[sadl.OpCode.Const] = sadl.decode.decodeOp(
  'Const',
  (ctx) => {
    const { cursor } = ctx;
    const version_num = parseInt(ctx.version, 10) || 0;

    // Dimensions (int32 count + int32 dims)
    const dims = sadl.decode.readCountedI32(cursor);

    // SADL0005+: optional sparsity + transposed flags
    let isSparse = 0;
    let sizeSparse = 0;
    let packedSparsitySize = 1;
    let transposed = false;
    if (version_num >= 5) {
      isSparse = cursor.readU8();
      if (isSparse !== 0) {
        sizeSparse = cursor.readI32();
        packedSparsitySize = cursor.readI16();
      }
      transposed = cursor.readU8() !== 0; // C++ bool (1 byte)
    }

    // TensorInternalType: Int32=0, Float=1, Int16=2
    const dtype = cursor.readI32();

    // Quantizer is present only for integer tensors
    const quantizer = (dtype === 0 || dtype === 2) ? cursor.readI32() : null;

    // Data payload (supports sparse tensors used in SADL0005+)
    let raw_data = null;
    let sparse = null;
    if (version_num >= 5 && isSparse !== 0) {
      const cols = (dims.length > 1) ? dims[1] : 0;
      const nbNonzerosColBytes = cols * 2; // uint16[cols]
      const indicesCount = packedSparsitySize ? Math.floor(sizeSparse / packedSparsitySize) : 0;
      const indicesBytes = indicesCount * 2; // uint16[indicesCount]

      const nbNonzerosCol = nbNonzerosColBytes ? cursor.readBytes(nbNonzerosColBytes) : null;
      const indices = indicesBytes ? cursor.readBytes(indicesBytes) : null;

      const dataBytes = sizeSparse * sadl.decode.dtypeSize(dtype);
      raw_data = dataBytes ? cursor.readBytes(dataBytes) : null;

      sparse = { sizeSparse, packedSparsitySize, nbNonzerosCol, indices };
    } else {
      const bytes = sadl.decode.product(dims) * sadl.decode.dtypeSize(dtype);
      raw_data = bytes ? cursor.readBytes(bytes) : null;
    }

    const attributes = [
      { name: 'dims', value: dims },
      { name: 'dtype', value: dtype },
      ...(quantizer !== null ? [{ name: 'quantizer', value: quantizer }] : []),
      ...(version_num >= 5 ? [
        { name: 'transposed', value: transposed },
        { name: 'isSparse', value: isSparse }
      ] : [])
    ];

    const const_data = { raw_data, dims, dtype, quantizer, transposed, isSparse, sparse };

    return {
      attributes,
      // Attach raw payload for initializer rendering / debugging.
      post: (node) => { node.const_data = const_data; }
    };
  },
  { input: [] }
);

// Placeholder
dec[sadl.OpCode.Placeholder] = sadl.decode.decodeOp('Placeholder', ({ cursor }) => {
  return [
    { name: 'dims', value: sadl.decode.readCountedI32(cursor) },
    { name: 'quantizer', value: cursor.readI32() }
  ];
}, { input: [] });

// Conv2D
dec[sadl.OpCode.Conv2D] = sadl.decode.decodeOp('Conv2D', ({ cursor }) => {
  return [
    { name: 'strides', value: sadl.decode.readCountedI32(cursor) },
    { name: 'pads', value: sadl.decode.readCountedI32(cursor) },
    { name: 'group', value: cursor.readI32() },
    { name: 'm_q', value: cursor.readI32() }
  ];
});

// Conv2DTranspose
dec[sadl.OpCode.Conv2DTranspose] = sadl.decode.decodeOp('Conv2DTranspose', ({ cursor }) => {
  return [
    { name: 'strides', value: sadl.decode.readCountedI32(cursor) },
    { name: 'pads', value: sadl.decode.readCountedI32(cursor) },
    { name: 'output_padding', value: sadl.decode.readCountedI32(cursor) },
    { name: 'm_q', value: cursor.readI32() }
  ];
});

// MatMul
dec[sadl.OpCode.MatMul] = sadl.decode.decodeOp('MatMul', ({ cursor }) => {
  return [{ name: 'm_q', value: cursor.readI32() }];
});

// Mul
dec[sadl.OpCode.Mul] = sadl.decode.decodeOp('Mul', ({ cursor }) => {
  return [{ name: 'm_q', value: cursor.readI32() }];
});

// MaxPool
dec[sadl.OpCode.MaxPool] = sadl.decode.decodeOp('MaxPool', ({ cursor }) => {
  return [
    { name: 'strides', value: sadl.decode.readCountedI32(cursor) },
    { name: 'kernel_shape', value: sadl.decode.readCountedI32(cursor) },
    { name: 'pads', value: sadl.decode.readCountedI32(cursor) }
  ];
});

// AveragePool
dec[sadl.OpCode.AveragePool] = sadl.decode.decodeOp('AveragePool', ({ cursor }) => {
  return [
    { name: 'strides', value: sadl.decode.readCountedI32(cursor) },
    { name: 'kernel_shape', value: sadl.decode.readCountedI32(cursor) },
    { name: 'pads', value: sadl.decode.readCountedI32(cursor) }
  ];
});

// Flatten
dec[sadl.OpCode.Flatten] = sadl.decode.decodeOp('Flatten', ({ cursor }) => {
  return [{ name: 'axis', value: cursor.readI32() }];
});

// GridSample
dec[sadl.OpCode.GridSample] = sadl.decode.decodeOp('GridSample', ({ cursor }) => {
  return [
    { name: 'align_corners', value: cursor.readI32() },
    { name: 'mode', value: cursor.readI32() },
    { name: 'padding_mode', value: cursor.readI32() }
  ];
});

// Resize
dec[sadl.OpCode.Resize] = sadl.decode.decodeOp('Resize', ({ cursor }) => {
  return [
    { name: 'input_label', value: cursor.readI32() },
    { name: 'coordinate_transformation_mode', value: cursor.readI32() },
    { name: 'mode', value: cursor.readI32() },
    { name: 'nearest_mode', value: cursor.readI32() }
  ];
});

// Compare (Less/Greater/Equal via mode)
dec[sadl.OpCode.Compare] = sadl.decode.decodeOp('Compare', ({ cursor }) => {
  return [{ name: 'mode', value: cursor.readI32() }]; // 0:Less, 1:Greater, 2:Equal
});

// ReduceMean
dec[sadl.OpCode.ReduceMean] = sadl.decode.decodeOp('ReduceMean', ({ cursor }) => {
  return [
    { name: 'axes', value: sadl.decode.readCountedI32(cursor) },
    { name: 'keepdims', value: cursor.readBytes(1)[0] !== 0 }
  ];
});

// Slice
dec[sadl.OpCode.Slice] = sadl.decode.decodeOp('Slice', ({ cursor }) => {
  return [
    { name: 'start_h', value: cursor.readI32() },
    { name: 'end_h', value: cursor.readI32() },
    { name: 'start_w', value: cursor.readI32() },
    { name: 'end_w', value: cursor.readI32() },
    { name: 'start_c', value: cursor.readI32() },
    { name: 'end_c', value: cursor.readI32() }
  ];
});

// Pad
dec[sadl.OpCode.Pad] = sadl.decode.decodeOp('Pad', ({ cursor }) => {
  return [
    { name: 'mode', value: cursor.readI32() }, // 0: constant, 1: reflect, 2: edge
    { name: 'pads', value: [cursor.readI32(), cursor.readI32(), cursor.readI32(), cursor.readI32()] }
  ];
});

// Depth2Space
dec[sadl.OpCode.Depth2Space] = sadl.decode.decodeOp('Depth2Space', ({ cursor }) => {
  return [
    { name: 'blocksize', value: cursor.readI32() }
  ];
});

// Gather
dec[sadl.OpCode.Gather] = sadl.decode.decodeOp('Gather', ({ cursor }) => {
  return [
    { name: 'axis', value: cursor.readI32() },
    { name: 'indices_is_scalar', value: cursor.readU8() !== 0 }
  ];
});

// GatherElements
dec[sadl.OpCode.GatherElements] = sadl.decode.decodeOp('GatherElements', ({ cursor }) => {
  return [
    { name: 'axis', value: cursor.readI32() }
  ];
});

// BitShift
dec[sadl.OpCode.BitShift] = sadl.decode.decodeOp('BitShift', ({ cursor }) => {
  return [
    { name: 'mode', value: cursor.readI32() } // 1: right shift, else: left shift
  ];
});

// passthrough (no payload)
const passthru = (type) => sadl.decode.decodeOp(type, () => []);

// register all decoders defined above
for (const [name, spec] of Object.entries(sadl.OpSpec)) {
  const id = spec.id;
  const fn = dec[id];

  if (fn) {
    sadl.registerOp(id, fn);
  } else if (spec.payload === false) {
    sadl.registerOp(id, passthru(name));
  }
}


// Placeholder MetaReader (maybe implemented later)
sadl.MetaReader = class {
  static async open(/* context */) { return null; }
};

sadl.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading SADL model.';
    }
};

export const ModelFactory = sadl.ModelFactory;
