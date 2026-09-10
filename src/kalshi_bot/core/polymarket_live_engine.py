from __future__ import annotations
_F='matched'
_E='no match'
_D='BUY'
_C=True
_B=False
_A=None
import asyncio,os,time
from dataclasses import dataclass,replace
from decimal import ROUND_CEILING,Decimal,InvalidOperation
from typing import Any
import httpx
from kalshi_bot.data.multi_crypto import MultiCryptoPriceFeed
from kalshi_bot.exchange.models import Side
from kalshi_bot.polymarket import PolymarketListing,PolymarketPublicClient
from kalshi_bot.strategies.examples.polymarket_momentum import PolymarketMomentumStrategy,SimSignal
from kalshi_bot.telemetry.logging import get_logger
logger=get_logger(__name__)
def _env_bool(name:str)->bool:return os.getenv(name,'').strip().lower()in{'1','true','yes','on'}
def _require(name:str)->str:
	value=os.getenv(name,'').strip()
	if not value:raise RuntimeError(f"Missing required deployment variable: {name}")
	return value
def _value(obj:Any,*keys:str,default:Any=_A)->Any:
	if isinstance(obj,dict):return next((obj[key]for key in keys if key in obj),default)
	return next((getattr(obj,key)for key in keys if hasattr(obj,key)),default)
def _fill_cents(response:Any,side:str,fallback:int)->int:
	try:
		making=Decimal(str(_value(response,'makingAmount','making_amount')));taking=Decimal(str(_value(response,'takingAmount','taking_amount')))
		if making<=0 or taking<=0:return fallback
		price=making/taking if side==_D else taking/making;return max(1,min(99,int((price*100).quantize(Decimal('1')))))
	except(InvalidOperation,TypeError,ValueError,ZeroDivisionError):return fallback
@dataclass(slots=_C)
class PendingLiveEntry:listing:PolymarketListing;signal:SimSignal;deadline:float;signed_order:Any|_A=_A;shares:int=0;uncertain:bool=_B
class PolymarketTradingClient:
	HOST='https://clob.polymarket.com';CHAIN_ID=137;SIGNATURE_TYPE=3
	def __init__(self)->_A:
		try:from py_clob_client_v2 import ApiCreds,ClobClient
		except ImportError as exc:raise RuntimeError('py-clob-client-v2 is not installed')from exc
		self.private_key=_require('POLYMARKET_SIGNER_PRIVATE_KEY');self.wallet=_require('POLYMARKET_WALLET_ADDRESS');self.host=os.getenv('POLYMARKET_CLOB_URL',self.HOST).rstrip('/');key=os.getenv('POLYMARKET_CLOB_API_KEY','').strip();secret=os.getenv('POLYMARKET_CLOB_SECRET','').strip();passphrase=os.getenv('POLYMARKET_CLOB_PASSPHRASE','').strip()
		if key and secret and passphrase:creds=ApiCreds(api_key=key,api_secret=secret,api_passphrase=passphrase)
		elif any((key,secret,passphrase)):raise RuntimeError('Set all three Polymarket CLOB credential variables or none')
		else:temporary=ClobClient(host=self.host,chain_id=self.CHAIN_ID,key=self.private_key,signature_type=self.SIGNATURE_TYPE,funder=self.wallet,use_server_time=_C);creds=temporary.create_or_derive_api_key()
		self._client=ClobClient(host=self.host,chain_id=self.CHAIN_ID,key=self.private_key,creds=creds,signature_type=self.SIGNATURE_TYPE,funder=self.wallet,use_server_time=_C,retry_on_error=_B)
	async def check_geoblock(self)->dict[str,Any]:
		async with httpx.AsyncClient(timeout=8.)as client:response=await client.get('https://polymarket.com/api/geoblock');response.raise_for_status();result=response.json()
		if not isinstance(result,dict):raise RuntimeError('Unexpected Polymarket geoblock response')
		return result
	async def collateral_balance(self)->float|_A:
		try:
			from py_clob_client_v2 import AssetType,BalanceAllowanceParams;result=await asyncio.to_thread(self._client.get_balance_allowance,BalanceAllowanceParams(asset_type=AssetType.COLLATERAL));raw=_value(result,'balance')
			if raw is _A:return
			amount=Decimal(str(raw));return float(amount/1000000 if amount>10000 else amount)
		except Exception:logger.exception('Could not read Polymarket collateral balance');return
	def _token(self,listing:PolymarketListing,side:Side)->str:
		token=listing.up_token_id if side is Side.YES else listing.down_token_id
		if not token:raise RuntimeError(f"No token id for {listing.slug} {side.value}")
		return token
	async def executable_buy_quote(self,listing:PolymarketListing,side:Side,*,slippage_cents:int)->tuple[int,int,float]|_A:
		book=await asyncio.to_thread(self._client.get_order_book,self._token(listing,side));levels:list[tuple[int,float]]=[]
		for row in _value(book,'asks',default=[])or[]:
			try:price=Decimal(str(_value(row,'price')));cents=int((price*100).to_integral_value(rounding=ROUND_CEILING));size=float(_value(row,'size',default=0))
			except(InvalidOperation,TypeError,ValueError):continue
			if 1<=cents<=99 and size>0:levels.append((cents,size))
		if not levels:return
		ask=min(price for(price,_)in levels);limit=min(99,ask+max(0,slippage_cents));depth=sum(size for(price,size)in levels if price<=limit);return ask,limit,depth
	async def build_limit_order(self,listing:PolymarketListing,side:Side,*,action:str,price_cents:int,shares:int)->Any:from py_clob_client_v2 import OrderArgs,PartialCreateOrderOptions,Side as ClobSide;args=OrderArgs(token_id=self._token(listing,side),price=price_cents/100,side=ClobSide.BUY if action==_D else ClobSide.SELL,size=shares);return await asyncio.to_thread(self._client.create_order,args,PartialCreateOrderOptions())
	async def post_fok(self,order:Any)->Any:from py_clob_client_v2 import OrderType;return await asyncio.to_thread(self._client.post_order,order,OrderType.FOK,_B,_B)
class PolymarketLiveEngine:
	def __init__(self,*,market_client:PolymarketPublicClient,trading_client:PolymarketTradingClient,feed:MultiCryptoPriceFeed,strategy:PolymarketMomentumStrategy,poll_interval:float=1.,execution_window_seconds:float=3.,live_enabled:bool=_B)->_A:self.market_client,self.trading_client=market_client,trading_client;self.feed,self.strategy=feed,strategy;self.poll_interval=max(.25,poll_interval);self.execution_window_seconds=max(.5,execution_window_seconds);self.live_enabled=live_enabled and _env_bool('POLY_LIVE');(self._targets):dict[str,float]={};(self._known):dict[str,PolymarketListing]={};(self._pending):dict[str,PendingLiveEntry]={};(self._exit_uncertain):set[str]=set()
	@staticmethod
	def _ok(response:Any)->bool:return bool(_value(response,'success','ok',default=_B))
	@staticmethod
	def _status(response:Any)->str:return str(_value(response,'status',default='')).lower()
	@staticmethod
	def _error(response:Any)->str:return str(_value(response,'errorMsg','error_msg',default='')or'')
	@classmethod
	def _no_fill(cls,response:Any)->bool:error,status=cls._error(response).lower(),cls._status(response);return not cls._ok(response)and(_E in error or'no orders found'in error or'fok'in error and'fill'in error or status=='unmatched')
	@staticmethod
	def _no_fill_exception(exc:Exception)->bool:message=str(exc).lower();return any(text in message for text in("couldn't be fully filled",'could not be fully filled','fully filled or killed',_E))
	def _target(self,listing:PolymarketListing)->float|_A:
		if listing.slug in self._targets:return self._targets[listing.slug]
		ticks=self.feed.snapshot(self.market_client.ASSETS[listing.asset])
		if not ticks:return
		opening=min(ticks,key=lambda x:abs(x.timestamp.timestamp()-listing.open_time))
		if abs(opening.timestamp.timestamp()-listing.open_time)>5:return
		self._targets[listing.slug]=opening.price;logger.warning('OPENING REFERENCE CAPTURED | ticker=%s | price=%.6f',listing.slug,opening.price);return opening.price
	async def _submit(self,pending:PendingLiveEntry,now:float)->bool:
		if now>=pending.deadline or pending.uncertain:return _C
		try:
			if pending.signed_order is _A:
				quote=await self.trading_client.executable_buy_quote(pending.listing,pending.signal.side,slippage_cents=self.strategy.entry_slippage_cents)
				if quote is _A:return _B
				ask,limit,depth=quote;pending.shares=max(self.strategy.contracts,(100+limit-1)//limit);cost=limit*pending.shares
				if self.strategy.reserved_cents()+cost>self.strategy.bankroll_cents:logger.warning('LIVE CANCEL | ticker=%s | reason=BANKROLL_CAP',pending.listing.slug);return _C
				if depth+1e-09<pending.shares:logger.warning('LIVE WAIT | ticker=%s | reason=INSUFFICIENT_DEPTH',pending.listing.slug);return _B
				pending.signal=replace(pending.signal,signal_ask=ask,limit_price=limit);pending.signed_order=await self.trading_client.build_limit_order(pending.listing,pending.signal.side,action=_D,price_cents=limit,shares=pending.shares)
			response=await self.trading_client.post_fok(pending.signed_order)
		except Exception as exc:
			if self._no_fill_exception(exc):pending.signed_order=_A;logger.warning('LIVE RETRY | ticker=%s | reason=FOK_NO_FILL',pending.listing.slug);return _B
			pending.uncertain=_C;logger.exception('ENTRY RESPONSE UNCERTAIN | ticker=%s',pending.listing.slug);return _C
		if self._ok(response)and self._status(response)==_F:fill=_fill_cents(response,_D,pending.signal.limit_price);self.strategy.open_position(pending.listing,pending.signal.side,fill,count=pending.shares,execution_mode='polymarket_live');logger.warning('POLYMARKET LIVE FILL | ticker=%s | fill=%dc | count=%d',pending.listing.slug,fill,pending.shares);return _C
		if self._no_fill(response):pending.signed_order=_A;return _B
		logger.error('ENTRY NOT CONFIRMED | ticker=%s | status=%s',pending.listing.slug,self._status(response));return _C
	async def _take_profit(self,listing:PolymarketListing,now:float)->_A:
		A='SELL';position=self.strategy.positions.get(listing.slug)
		if position is _A or listing.slug in self._exit_uncertain:return
		bid=listing.yes_bid if position.side is Side.YES else listing.no_bid
		if bid is _A or bid<self.strategy.take_profit:return
		try:order=await self.trading_client.build_limit_order(listing,position.side,action=A,price_cents=self.strategy.take_profit,shares=position.count);response=await self.trading_client.post_fok(order)
		except Exception as exc:
			if self._no_fill_exception(exc):return
			self._exit_uncertain.add(listing.slug);logger.exception('EXIT RESPONSE UNCERTAIN | ticker=%s',listing.slug);return
		if self._ok(response)and self._status(response)==_F:fill=_fill_cents(response,A,self.strategy.take_profit);self.strategy.close_position(listing.slug,fill,'TAKE_PROFIT_98',now);logger.warning('POLYMARKET LIVE EXIT | ticker=%s | fill=%dc',listing.slug,fill)
	async def _settle_missing(self,active:set[str],now:float)->_A:
		for slug in set(self.strategy.positions)-active:
			listing=self._known.get(slug)
			if listing is _A or now<listing.close_time:continue
			resolved=await self.market_client.resolved(listing)
			if resolved is _A or resolved.resolved_side not in{'yes','no'}:continue
			position=self.strategy.positions.get(slug)
			if position is _A:continue
			won=position.side.value==resolved.resolved_side;self.strategy.close_position(slug,100 if won else 0,'SETTLEMENT_WIN'if won else'SETTLEMENT_LOSS',now);self._exit_uncertain.discard(slug)
	async def preflight(self)->_A:
		if not self.live_enabled:raise RuntimeError('Start with --live and set POLY_LIVE=true')
		geo=await self.trading_client.check_geoblock()
		if geo.get('blocked'):raise RuntimeError(f"Server is geoblocked: {geo.get("country")} / {geo.get("region")}")
		balance=await self.trading_client.collateral_balance();logger.warning('POLYMARKET PREFLIGHT OK | collateral=%s','unknown'if balance is _A else f"${balance:.2f}")
	async def run(self,max_cycles:int|_A=_A)->_A:
		await self.preflight();logger.warning('POLYMARKET LIVE STARTED | LIVE_ORDERS=ENABLED');await self.feed.start();cycle=0
		try:
			while max_cycles is _A or cycle<max_cycles:
				try:
					now=time.time();listings=await self.market_client.get_open_crypto_markets(now);listings=[x for x in listings if x.up_token_id and x.down_token_id];self._known.update({x.slug:x for x in listings});await self._settle_missing({x.slug for x in listings},now)
					for listing in listings:
						await self._take_profit(listing,now);pending=self._pending.get(listing.slug)
						if pending:
							if await self._submit(pending,now):self._pending.pop(listing.slug,_A)
							continue
						if listing.slug in self.strategy.positions:continue
						product=self.market_client.ASSETS[listing.asset];signal=self.strategy.evaluate(listing,self._target(listing),now,self.feed.snapshot(product))
						if signal:
							pending=PendingLiveEntry(listing,signal,now+self.execution_window_seconds);self._pending[listing.slug]=pending
							if await self._submit(pending,now):self._pending.pop(listing.slug,_A)
					keep={x.slug for x in listings}|set(self.strategy.positions)|set(self._pending);self.strategy.prune(keep);self._targets={key:value for(key,value)in self._targets.items()if key in keep}
				except Exception:logger.exception('POLYMARKET LIVE SCAN FAILED; retrying')
				cycle+=1
				if max_cycles is _A or cycle<max_cycles:await asyncio.sleep(self.poll_interval)
		finally:await self.feed.stop()
