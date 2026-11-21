import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Streamlit, withStreamlitConnection} from 'streamlit-component-lib';
import {
  ReactFlow, ReactFlowProvider, Handle, Position, Background, Controls, BackgroundVariant,
  applyNodeChanges, applyEdgeChanges, useUpdateNodeInternals,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {BLOCKS, EDGE_MARKER, EDGE_STYLE, graphFromPython,
        makeEdge, makeNode, initialPosition, replaceConnection} from './graphState.js';
import './styles.css';

function PipelineNode({id,data,selected}){
  const orientation=data.orientation||'horizontal';
  const updateNodeInternals=useUpdateNodeInternals();
  const {removeNode}=React.useContext(EditorActionsContext);
  useEffect(()=>updateNodeInternals(id),[id,orientation,updateNodeInternals]);
  return <div className={'pipeline-node'+(selected?' active':'')}>
    <Handle type="target" position={orientation==='vertical'?Position.Top:Position.Left}
            id="in" className="pipeline-handle"/>
    <span>{data.label}</span>
    <button className="node-delete-button nodrag nopan" type="button"
      aria-label={`Удалить блок ${data.label}`} title="Удалить блок"
      onClick={event=>{event.stopPropagation();removeNode(id);}}>×</button>
    <Handle type="source" position={orientation==='vertical'?Position.Bottom:Position.Right}
            id="out" className="pipeline-handle"/>
  </div>;
}
const nodeTypes={pipeline:PipelineNode};
const edgeOptions={type:'smoothstep',style:EDGE_STYLE,markerEnd:EDGE_MARKER,selectable:true,deletable:true,reconnectable:true};
const EditorActionsContext=React.createContext({removeNode:()=>{}});

function Editor({args}){
  const orientation=args.orientation==='vertical'?'vertical':'horizontal';
  const [nodes,setNodes]=useState([]);
  const [edges,setEdges]=useState([]);
  const [drawerOpen,setDrawerOpen]=useState(true);
  const [edgeSelected,setEdgeSelected]=useState(null);
  const [nodeSelected,setNodeSelected]=useState(null);
  const [sourceChoice,setSourceChoice]=useState('');
  const [targetChoice,setTargetChoice]=useState('');
  const [notice,setNotice]=useState('');
  const seqRef=useRef(0);
  const instanceRef=useRef(`editor-${Date.now()}-${Math.random()}`);
  const revisionRef=useRef(null);

  const emit=useCallback((type,payload)=>{
    Streamlit.setComponentValue({event:{instance:instanceRef.current,seq:++seqRef.current,type,payload}});
  },[]);

  // Deliberately sync only when Python changes the actual graph. Ordinary
  // Streamlit reruns and opening/closing the library never reset zoom or nodes.
  useEffect(()=>{
    if(revisionRef.current===args.graphRevision) return;
    revisionRef.current=args.graphRevision;
    const next=graphFromPython(args.graph,orientation);
    setNodes(next.nodes);
    setEdges(next.edges);
    setEdgeSelected(null);
    setNodeSelected(null);
    setNotice('');
  },[args.graphRevision,args.graph,orientation]);

  useEffect(()=>{
    Streamlit.setFrameHeight(585);
  },[]);

  const onNodesChange=useCallback(changes=>{
    setNodes(current=>applyNodeChanges(changes,current));
  },[]);
  const onEdgesChange=useCallback(changes=>{
    setEdges(current=>applyEdgeChanges(changes,current));
  },[]);

  const addBlock=useCallback(id=>{
    if(nodes.some(n=>n.id===id)) return;
    const position=initialPosition(id,orientation);
    setNodes(current=>[...current,makeNode(id,position,orientation)]);
    emit('add_node',{id});
  },[nodes,orientation,emit]);

  const removeEdge=useCallback(edge=>{
    if(!edge) return;
    setEdges(current=>current.filter(e=>e.id!==edge.id));
    setEdgeSelected(null);
    emit('disconnect',{source:edge.source,target:edge.target});
  },[emit]);

  const removeNode=useCallback(id=>{
    if(!id) return;
    setNodes(current=>current.filter(node=>node.id!==id));
    setEdges(current=>current.filter(edge=>edge.source!==id&&edge.target!==id));
    setNodeSelected(null);
    setEdgeSelected(null);
    setNotice('Блок и связанные с ним стрелки удалены.');
    emit('remove_node',{id});
  },[emit]);

  const onConnect=useCallback(connection=>{
    const {source,target}=connection;
    if(!source||!target||source===target) return;
    setEdges(current=>replaceConnection(current,source,target));
    setNotice('Связь создана. При необходимости прежние конфликтующие соединения заменены.');
    emit('connect',{source,target});
  },[emit]);

  const onReconnect=useCallback((oldEdge,newConnection)=>{
    const {source,target}=newConnection;
    if(!source||!target||source===target) return;
    setEdges(current=>replaceConnection(current,source,target,oldEdge.id));
    setEdgeSelected(null);
    setNotice('Стрелка перенаправлена.');
    emit('reconnect',{oldSource:oldEdge.source,oldTarget:oldEdge.target,source,target});
  },[emit]);

  // One atomic event for a mixed selection of nodes and edges.
  // Multiple setComponentValue calls during one keyboard action would race.
  const onDelete=useCallback(({nodes:deletedNodes,edges:deletedEdges})=>{
    const nodeIds=(deletedNodes||[]).map(n=>n.id);
    const links=(deletedEdges||[]).map(e=>({source:e.source,target:e.target}));
    if(nodeIds.length||links.length){
      emit('delete_elements',{nodeIds,links});
    }
    setEdgeSelected(null);
    setNodeSelected(null);
  },[emit]);

  const currentEdge=useMemo(()=>edges.find(e=>e.id===edgeSelected)||null,[edges,edgeSelected]);
  const currentNode=useMemo(()=>nodes.find(n=>n.id===nodeSelected)||null,[nodes,nodeSelected]);
  const choices=useMemo(()=>nodes.map(n=>({id:n.id,label:BLOCKS.find(b=>b.id===n.id)?.title||n.id})),[nodes]);

  const changeConnection=()=>{
    if(!currentEdge||!sourceChoice||!targetChoice||sourceChoice===targetChoice){
      setNotice('Выберите два разных блока для соединения.'); return;
    }
    onReconnect(currentEdge,{source:sourceChoice,target:targetChoice});
  };

  const selectEdge=(edge)=>{
    setEdgeSelected(edge.id);
    setNodeSelected(null);
    setSourceChoice(edge.source);
    setTargetChoice(edge.target);
  };

  return <div className="editor-root">
    <aside className={'editor-drawer'+(drawerOpen?' open':' closed')} aria-label="Библиотека блоков">
      {drawerOpen && <div className="drawer-inner">
        <div className="drawer-title">Библиотека блоков</div>
        <p className="drawer-explain">Добавьте этап, затем соедините его с другими.</p>
        {BLOCKS.map(b=>{
          const exists=nodes.some(n=>n.id===b.id);
          return <button className="block-option" key={b.id} onClick={()=>addBlock(b.id)} disabled={exists}>
            <strong>{b.title}</strong><small>{exists?'На схеме':b.hint}</small>
          </button>;
        })}
      </div>}
    </aside>
    <button className="drawer-toggle" type="button"
       title={drawerOpen?'Скрыть библиотеку':'Открыть библиотеку'}
       aria-label={drawerOpen?'Скрыть библиотеку':'Открыть библиотеку'}
       aria-expanded={drawerOpen} onClick={()=>setDrawerOpen(v=>!v)}>{drawerOpen?'❮':'❯'}</button>
    <section className="editor-canvas" aria-label="Рабочее поле пайплайна">
      <EditorActionsContext.Provider value={{removeNode}}><ReactFlow
        nodes={nodes} edges={edges} nodeTypes={nodeTypes}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
        onConnect={onConnect} onReconnect={onReconnect}
        onDelete={onDelete}
        onNodeDragStop={(_,n)=>emit('move_node',{id:n.id,position:n.position})}
        onNodeClick={(_,n)=>{setEdgeSelected(null);setNodeSelected(n.id);}}
        onEdgeClick={(_,e)=>selectEdge(e)}
        onEdgeContextMenu={(event,e)=>{event.preventDefault();selectEdge(e);}}
        onPaneClick={()=>{setEdgeSelected(null);setNodeSelected(null);}}
        isValidConnection={conn=>!!conn.source&&!!conn.target&&conn.source!==conn.target}
        defaultEdgeOptions={edgeOptions}
        edgesReconnectable={true} reconnectRadius={24}
        connectionRadius={24} snapToGrid={true} snapGrid={[10,10]}
        deleteKeyCode={['Backspace','Delete']} fitView={true} fitViewOptions={{padding:.25}}
        minZoom={0.5} maxZoom={1.8} panOnDrag={true}
        proOptions={{hideAttribution:true}}>
        <Background variant={BackgroundVariant.Dots} gap={20} size={1.2} color="#B6C9C4"/>
        <Controls showInteractive={false}/>
      </ReactFlow></EditorActionsContext.Provider>
      <div className="editor-toolbar" role="status">{notice||'Нажмите на линию для её удаления или перенаправления. Концы стрелок можно перетаскивать мышью.'}</div>
      {currentNode && <div className="node-toolbar" role="group" aria-label="Управление выбранным блоком">
        <b>{BLOCKS.find(block=>block.id===currentNode.id)?.title||currentNode.id}</b>
        <button onClick={()=>emit('select_node',{id:currentNode.id})} className="apply" title="Открыть настройки выбранного блока">Открыть настройки</button>
        <button onClick={()=>removeNode(currentNode.id)} className="remove" title="Удалить выбранный блок и его стрелки">Удалить блок</button>
        <button onClick={()=>setNodeSelected(null)} aria-label="Закрыть управление блоком">×</button>
      </div>}
      {currentEdge && <div className="edge-toolbar" role="group" aria-label="Управление выбранной стрелкой">
        <b>Стрелка</b>
        <label>Откуда<select value={sourceChoice} onChange={e=>setSourceChoice(e.target.value)}>
          {choices.map(c=><option key={c.id} value={c.id}>{c.label}</option>)}
        </select></label>
        <label>Куда<select value={targetChoice} onChange={e=>setTargetChoice(e.target.value)}>
          {choices.map(c=><option key={c.id} value={c.id}>{c.label}</option>)}
        </select></label>
        <button onClick={changeConnection} className="apply" title="Изменить связь">Перенаправить</button>
        <button onClick={()=>removeEdge(currentEdge)} className="remove" title="Удалить связь">Открепить</button>
        <button onClick={()=>setEdgeSelected(null)} aria-label="Закрыть управление стрелкой">×</button>
      </div>}
    </section>
  </div>;
}

function ConnectedEditor(props){return <ReactFlowProvider><Editor args={props.args}/></ReactFlowProvider>;}
const Component=withStreamlitConnection(ConnectedEditor);
createRoot(document.getElementById('root')).render(<Component/>);

