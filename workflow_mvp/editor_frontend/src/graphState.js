/** One canonical shape for all connections (manual, restored, newly drawn). */
export const BLOCKS = [
  {id:'input',title:'1. Входные данные',hint:'CSV и характеристики таблицы'},
  {id:'preprocess',title:'2. Подготовка данных',hint:'Признаки, пропуски, Train/Test'},
  {id:'analysis',title:'3. Анализ',hint:'Метод и его параметры'},
  {id:'output',title:'4. Вывод',hint:'Метрики, графики, Word-отчёт'},
];
export const EDGE_STYLE = {stroke:'#256D73', strokeWidth:3};
export const EDGE_MARKER = {type:'arrowclosed',color:'#256D73',width:21,height:21};
export const edgeId = (a,b)=>`edge:${a}:${b}`;
export function makeEdge(source,target){
  return {id:edgeId(source,target),source,target,type:'smoothstep',style:{...EDGE_STYLE},
          markerEnd:{...EDGE_MARKER},reconnectable:true,selectable:true,deletable:true};
}
export function makeNode(id,position,orientation){
  return {id,type:'pipeline',data:{label:BLOCKS.find(b=>b.id===id)?.title||id,orientation},
          position:{x:Number(position.x),y:Number(position.y)},draggable:true,selectable:true};
}
export function initialPosition(id,orientation){
  const i=BLOCKS.findIndex(b=>b.id===id);
  return orientation==='vertical'?{x:180,y:i*162}:{x:i*265,y:165};
}
export function replaceConnection(edges, source, target, oldEdgeId=null){
  if(!source||!target||source===target) return edges;
  const rest=edges.filter(e=>e.id!==oldEdgeId && e.source!==source && e.target!==target);
  return [...rest,makeEdge(source,target)];
}
export function graphFromPython(graph,orientation){
  return {
    nodes:(graph?.nodes||[]).map(n=>makeNode(n.id,n.position,orientation)),
    edges:(graph?.edges||[]).map(e=>makeEdge(e.source,e.target))
  };
}

